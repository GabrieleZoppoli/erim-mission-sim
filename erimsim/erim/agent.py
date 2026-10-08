"""The ERIM agent: ensembles of MLP estimators on the moving window (problems P_x, P_z, P_f, P_g), a CNN recogniser
with Bayesian accumulation (P_X), and an MLP policy (C_t) acting on the estimates under certainty equivalence.
The online quality proxies are the ensemble disagreements (trace of the ensemble covariance of each estimate) and
1 - max posterior for recognition."""
from typing import NamedTuple
import jax
import jax.numpy as jnp
from .. import quat, windows, models, render
from ..sim import Summary


class Norm(NamedTuple):
    f_mean: jnp.ndarray
    f_std: jnp.ndarray
    tx_mean: jnp.ndarray
    tx_std: jnp.ndarray
    tz_mean: jnp.ndarray
    tz_std: jnp.ndarray
    tf_mean: jnp.ndarray
    tf_std: jnp.ndarray
    tg_mean: jnp.ndarray
    tg_std: jnp.ndarray


class ErimParams(NamedTuple):
    est_x: list          # ensemble (leading axis E) of MLP params
    est_z: list
    est_f: list
    est_g: list
    cnn: dict
    pol: list
    norm: Norm


class ErimState(NamedTuple):
    win: jnp.ndarray
    log_post: jnp.ndarray


POL_IN = 3 + 3 + 3 + 3 + 6 + 4 + 2 + render.N_CLASSES      # 32


def init_params(key, cfg, mcfg, norm, n_hidden=None, cnn_J=None):
    n_hidden = n_hidden or mcfg.n_hidden
    cnn_J = cnn_J or mcfg.cnn_J
    k = jax.random.split(key, 6)
    n_in = (mcfg.N + 1) * windows.FEAT
    hidden = [n_hidden] * mcfg.depth
    mk = lambda kk, n_out: models.init_mlp(kk, [n_in] + hidden + [n_out])
    return ErimParams(
        est_x=models.init_ensemble(k[0], mcfg.ensemble, lambda kk: mk(kk, windows.DIM_TX)),
        est_z=models.init_ensemble(k[1], mcfg.ensemble, lambda kk: mk(kk, windows.DIM_TZ)),
        est_f=models.init_ensemble(k[2], mcfg.ensemble, lambda kk: mk(kk, windows.DIM_TF)),
        est_g=models.init_ensemble(k[3], mcfg.ensemble, lambda kk: mk(kk, windows.DIM_TG)),
        cnn=models.init_cnn(k[4], cnn_J, mcfg.cnn_ch, mcfg.cnn_fc, render.N_CLASSES),
        pol=models.init_mlp(k[5], [POL_IN, mcfg.pol_hidden, mcfg.pol_hidden, 6], scale=0.5),
        norm=norm)


def estimate(ens, feats_n, mean, std):
    """Ensemble mean (de-standardised) and disagreement = trace of the ensemble covariance (standardised units)."""
    out = models.ensemble_apply(models.mlp, ens, feats_n)      # [E, d]
    mu = jnp.mean(out, 0)
    dis = jnp.sum(jnp.var(out, 0))
    return mu * std + mean, dis


def estimates(params, win):
    feats, anchor = windows.features(win)
    fn = (feats - params.norm.f_mean) / params.norm.f_std
    tx, dx = estimate(params.est_x, fn, params.norm.tx_mean, params.norm.tx_std)
    tz, dz = estimate(params.est_z, fn, params.norm.tz_mean, params.norm.tz_std)
    tf, df = estimate(params.est_f, fn, params.norm.tf_mean, params.norm.tf_std)
    tg, dg = estimate(params.est_g, fn, params.norm.tg_mean, params.norm.tg_std)
    p, cols, v, w = windows.unpack_tx(tx, anchor)
    q = quat.from_two_columns(cols[0:3], cols[3:6])
    x_hat = jnp.concatenate([p, q, v, w])
    z_hat = jnp.concatenate([p + tz[0:3], tz[3:6]])
    return x_hat, z_hat, tf, tg, jnp.array([dx, dz, df, dg]), cols


def policy_input(x_hat, z_hat, f_hat, g_hat, cols, log_post, use_post):
    post = jnp.exp(log_post) * use_post
    return jnp.concatenate([z_hat[0:3] - x_hat[0:3], z_hat[3:6] - x_hat[7:10], x_hat[7:10], x_hat[10:13],
                            cols, f_hat, g_hat, post])


class ErimAgent:
    """Class attributes set by bind(): cfg, mcfg, use_images (CNN updates the posterior), use_post (policy sees it)."""
    use_images = True
    use_post = False

    @classmethod
    def init(cls, params, obs0, cfg):
        y_own, y_obj, img = obs0
        win = windows.init_window(cls.mcfg.N, windows.row(y_own, y_obj, jnp.zeros(6)))
        return ErimState(win, jnp.full(render.N_CLASSES, -jnp.log(render.N_CLASSES)))

    @classmethod
    def update(cls, params, S, obs, u_prev, t):
        y_own, y_obj, img = obs
        win = windows.push(S.win, windows.row(y_own, y_obj, u_prev))
        x_hat, z_hat, f_hat, g_hat, dis, cols = estimates(params, win)
        if cls.use_images:
            ll = jax.nn.log_softmax(models.cnn(params.cnn, img))
            lp = S.log_post + ll
            lp = lp - jax.scipy.special.logsumexp(lp)
        else:
            lp = S.log_post
        sm = Summary(x_hat=x_hat, z_hat=z_hat, f_hat=f_hat, g_hat=g_hat, log_post=lp,
                     proxies=jnp.concatenate([dis, jnp.array([1.0 - jnp.exp(jnp.max(lp))])]))
        return ErimState(win, lp), sm

    @classmethod
    def act(cls, params, sm, obs, t, key):
        _, _, _, _, _, cols = None, None, None, None, None, windows.att_cols(sm.x_hat[3:7])
        inp = policy_input(sm.x_hat, sm.z_hat, sm.f_hat, sm.g_hat, cols, sm.log_post, 1.0 if cls.use_post else 0.0)
        a = models.mlp(params.pol, inp)
        cfg = cls.cfg
        return jnp.concatenate([cfg.u_max * jnp.tanh(a[0:3]), cfg.w_max * jnp.tanh(a[3:6])])


def bind(cfg, mcfg, use_images=True, use_post=False):
    class A(ErimAgent):
        pass
    A.cfg, A.mcfg, A.use_images, A.use_post = cfg, mcfg, use_images, use_post
    return A

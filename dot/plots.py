import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib import animation
from corner import corner as dfm_corner
import pymc3 as pm

__all__ = ['corner', 'posterior_predictive', 'movie']


def corner(trace, **kwargs):
    """
    Make a corner plot
    """
    return dfm_corner(pm.trace_to_dataframe(trace), **kwargs)


def posterior_predictive(model, trace, samples=100, path=None, **kwargs):
    """
    Take draws from the posterior predictive given a trace and a model.
    """
    with model:
        ppc = pm.sample_posterior_predictive(trace, samples=samples,
                                             **kwargs)

    fig, ax = plt.subplots(figsize=(10, 2.5))
    plt.errorbar(model.lc.time,
                 model.lc.flux,
                 model.lc.flux_err,
                 fmt='.', color='k', ecolor='silver')

    plt.plot(model.lc.time[model.mask][::model.skip_n_points],
             ppc[f'{model.n_spots}_obs'].T,
             color='DodgerBlue', lw=2, alpha=0.1)

    plt.gca().set(xlabel='Time [d]', ylabel='Flux',
                  xlim=[model.lc.time[model.mask].min(),
                        model.lc.time[model.mask].max()])
    if path is not None:
        fig.savefig(path, bbox_inches='tight')
    return fig, ax


def posterior_shear(model, trace, path=None):
    """
    Plot the posterior distribution for the stellar differential rotation shear.
    """
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hist(trace[f'{model.n_spots}_shear'],
            bins=25,
            range=[0, 0.6],
            color='k')
    ax.set(xlabel='Shear $\\alpha$')
    for sp in ['right', 'top']:
        ax.spines[sp].set_visible(False)
    if path is not None:
        fig.savefig(path, bbox_inches='tight')
    return fig, ax


def movie(results_dir, model, trace, xsize=250, fps=10,
          artifical_photometry=False, posterior_samples=10,
          dpi=250, u1=0.4, u2=0.2):
    """
    Plot an animation of the light curve and the rotating stellar surface.
    """
    # Get median parameter values for system setup:
    n_spots = model.n_spots
    shear = np.median(trace[f'{n_spots}_shear'])
    complement_to_inclination = np.median(trace[f'{n_spots}_comp_inc'])
    eq_period = np.median(trace[f'{n_spots}_P_eq'])

    samples = pm.trace_to_dataframe(trace).values
    if isinstance(model.contrast, float):
        parameters_per_spot = 3
        contrast = model.contrast
    else:
        # TODO: implement floating contrasts
        raise NotImplementedError('TODO: Need to implement floating contrast '
                                  'model')

    spot_props = np.median(samples[:, 4:], axis=0).reshape((n_spots,
                                                            parameters_per_spot))

    # Define the spot properties
    spot_lons = spot_props[:, 0]
    spot_lats = spot_props[:, 1]
    spot_rads = spot_props[:, 2]

    # Create grid of pixels on which we will pixelate the spotted star:
    xgrid = np.linspace(-1, 1, xsize)
    xx, yy = np.meshgrid(xgrid, xgrid)
    m = np.zeros((xsize, xsize,
                  len(model.lc.time[model.mask][::model.skip_n_points])))

    # Compute a simple, artistic limb-darkening factor
    r = np.hypot(xx, yy)
    ld = (1 - u1 * r ** 2 - u2 * r) / (1 - u1 / 3 - u2 / 6)

    # Compute a mask for pixels that fall on the star
    on_star = np.hypot(xx, yy) <= 1
    # Set the pixel intensities to the limb-darkened values
    m[..., :] = (ld * on_star.astype(int))[..., None]

    # For each spot:
    spot_model = 1
    for spot_ind in range(n_spots):
        # Make everything spin
        period_i = eq_period / (1 - shear * np.sin(spot_lats[spot_ind] - np.pi / 2) ** 2)
        phi = 2 * np.pi / period_i * (model.lc.time[model.mask][::model.skip_n_points] -
                                      model.lc.time[model.mask].mean()) - spot_lons[spot_ind]

        # Compute the spot position as a function of time:
        spot_position_x = (np.cos(phi - np.pi / 2) * np.sin(complement_to_inclination) *
                           np.sin(spot_lats[spot_ind]) +
                           np.cos(complement_to_inclination) * np.cos(spot_lats[spot_ind]))
        spot_position_y = -np.sin(phi - np.pi / 2) * np.sin(spot_lats[spot_ind])
        spot_position_z = (np.cos(spot_lats[spot_ind]) * np.sin(complement_to_inclination) -
                           np.sin(phi) * np.cos(complement_to_inclination) * np.sin(spot_lats[spot_ind]))

        # Compute the distance from the spot to the center of the stellar disk
        rsq = spot_position_x ** 2 + spot_position_y ** 2

        # Foreshorten the spots as they approach the limb,
        # mask the spots that land on the opposite stellar hemisphere
        spot_model -= (spot_rads[spot_ind] ** 2 * (1 - model.contrast) *
                       np.where(spot_position_z > 0, np.sqrt(1 - rsq), 0))

        foreshorten_semiminor_axis = np.sqrt(1 - rsq)

        # Compute pixels that fall on a spot:
        a = spot_rads[spot_ind]
        # Semi-minor axis
        b = spot_rads[spot_ind] * foreshorten_semiminor_axis
        # Semi-major axis rotation
        A = np.pi / 2 + np.arctan2(spot_position_y, spot_position_x)[None, None, :]
        on_spot = (((xx[:, :, None] - spot_position_x[None, None, :]) * np.cos(A) +
                    (yy[:, :, None] - spot_position_y[None, None, :]) * np.sin(A)) ** 2 / a ** 2 +
                   ((xx[:, :, None] - spot_position_x[None, None, :]) * np.sin(A) -
                    (yy[:, :, None] - spot_position_y[None, None, :]) * np.cos(A)) ** 2 / b ** 2 <= 1)
        on_spot *= spot_position_z[None, None, :] > 0
        m[on_spot] *= 1 - contrast

    # Draw samples from the posterior in the light curve domain
    with model:
        ppc = pm.sample_posterior_predictive(trace, samples=posterior_samples)

    print(f"Generating animation with {m.shape[2]} frames:")
    gs = GridSpec(1, 5)

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure(
        figsize=(7, 2),
        dpi=dpi
    )
    gs = GridSpec(1, 5, figure=fig)

    ax_image = plt.subplot(gs[0:2])
    # Plot the initial image (first frame)
    im = ax_image.imshow(m[..., 0],
             aspect='equal',
             cmap=plt.cm.copper,
             extent=[-1, 1, -1, 1],
             vmin=0,
             vmax=1,
             origin='lower'
         )
    ax_image.axis('off')

    # Plot the light curve
    ax_lc = plt.subplot(gs[2:])
    ax_lc.plot(model.lc.time[model.mask][::model.skip_n_points],
               ppc[f'{n_spots}_obs'].T,
               color='DodgerBlue', alpha=0.05)
    ax_lc.plot(model.lc.time[model.mask][::model.skip_n_points],
               model.lc.flux[model.mask][::model.skip_n_points],
               '.', color='k')

    # Optionally plot the "synthetic photometry" of the pixelated visualization
    if artifical_photometry:
        artifical_lc = m.sum(axis=(0, 1))
        ax_lc.plot(model.lc.time[model.mask][::model.skip_n_points],
                   artifical_lc/artifical_lc.mean(), color='r')

    ax_lc.set(xlabel='Time', ylabel='Flux')

    for sp in ['right', 'top']:
        ax_lc.spines[sp].set_visible(False)
    time_marker = ax_lc.axvline(model.lc.time[model.mask][::model.skip_n_points][0],
                                ls='--', color='gray', zorder=-10)

    def animate_func(ii):
        if ii % fps == 0:
            print('.', end='')

        im.set_array(np.ma.masked_array(m[..., ii].T, m[..., ii].T == 0))

        time_marker.set_data([model.lc.time[model.mask][::model.skip_n_points][ii],
                              model.lc.time[model.mask][::model.skip_n_points][ii]],
                             [0, 1])

        return [im]

    fig.tight_layout()
    anim = animation.FuncAnimation(
        fig,
        animate_func,
        frames=m.shape[2],
        interval=1000 / fps,  # in milliseconds
    )

    anim.save(os.path.join(results_dir, 'movie.mp4'),
              fps=fps, extra_args=['-vcodec', 'libx264'])
    print('done')

    return fig, m
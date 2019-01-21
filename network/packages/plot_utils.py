import numpy as np
from scipy import interpolate




def plot_strains(ax1, strain_base, strain_preds, strain_err, strain_idx, strain_label, y_axis_limit, es_frame_idx=0 ):
    x = np.arange(1,21)

    y = strain_base[:,strain_idx]
    # print('y', y.shape)
    x2 = np.linspace(x[0], x[-1], 100)
    y2 = interpolate.pchip_interpolate(x, y, x2)

    ax1.axis([1, 20, -y_axis_limit, y_axis_limit])
    ax1.axhline(0, color='black')
    ax1.axvline(es_frame_idx+1, color='gray', linestyle='--')
    ax1.set_xticks(np.arange(1, 20, step=2))

    ax1.set(xlabel='frame', ylabel='strain')
    ax1.plot(x2, y2, label='ground truth {}'.format(strain_label), color='red')
    ax1.plot(x, y, "o", color='red')

    # -- 
    x = np.arange(1,21)
    y = strain_preds[:,strain_idx]
    # print('y', y.shape)
    x2 = np.linspace(x[0], x[-1], 100)
    y2 = interpolate.pchip_interpolate(x, y, x2)

    ax1.plot(x2, y2, label='prediction {}'.format(strain_label), color='green')
    ax1.plot(x, y, "o", color='green')

    # --
    x = np.arange(1,21)
    y = strain_err[:,strain_idx]
    # print('y', y.shape)
    x2 = np.linspace(x[0], x[-1], 100)
    y2 = interpolate.pchip_interpolate(x, y, x2)

    err_color = 'purple'
    ax1.plot(x2, y2, label='{} error'.format(strain_label), color=err_color)
    ax1.plot(x, y, "o", color=err_color)


    ax1.legend()

def plot_strains_and_displacements(strain_base, strain_preds, strain_err, disp_err):
    fig, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2,2, figsize=(15,15))
    fig.set_tight_layout(True)

    strain_idx = 9
    plot_strains(ax1, strain_base, strain_preds, strain_err, strain_idx, 'CC', 0.2)

    strain_idx = 4
    plot_strains(ax2, strain_base, strain_preds, strain_err, strain_idx, 'RR', 0.4)

    strain_idx = 4
    x = np.arange(20)

    y = disp_err
    # print('y', y.shape)
    x2 = np.linspace(x[0], x[-1], 100)
    y2 = interpolate.pchip_interpolate(x, y, x2)

    ax3.plot(x2, y2, label='displacement err')
    ax3.plot(x, y, "o")
    ax3.axhline(0, color='black')
    ax3.set_xticks(np.arange(0, 19, step=2))
    ax3.legend()

    
    mng = plt.get_current_fig_manager()
    #mng.frame.Maximize(True)
    mng.window.showMaximized()

    plt.show()
    plt.close()
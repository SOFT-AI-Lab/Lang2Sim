import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def LDFT2D_MT(M, T):
    """compute water adsorption isotherm of a hydrophilic porous matrix at different temperatures."""

    #grids=np.float32(np.random.randint(2, size=(1, 20,20)))
    grids=M

    batch_size = grids.shape[0]
    GRID_SIZE = grids.shape[1]
    
    inner_loops=100

    N_ITER = 80
    N_ADSORP = 40
    STEP_SIZE = 0.025
    N_SQUARES = GRID_SIZE**2

    # Physical constants
    #T = 298.0                  # temperature K
    Y = 1.5
    #Y = Interaction
    TC = 647.0                 # K  # critical temperature
    KB = 0.0019872041          # kcal/(mol?K)  # boltsman constant
    BETA = 1/(KB*T)
    MUSAT = -2.0 * KB * TC     # saturation chemical potential
    C = 4.0                    # kcal/(mol?K)  # square is 4, triangle is 3.
    WFF = -2.0 * MUSAT/C       # water-water interaction   # interaction energy
    WMF = Y * WFF              # water-matrix interaction  # interaction energy


    muu_lookup = list()
    for jj in range(N_ITER + 1):
        if jj <= N_ADSORP:
            RH = jj * STEP_SIZE
        else:
            RH = N_ADSORP*STEP_SIZE - (jj-N_ADSORP)*STEP_SIZE
        if RH == 0:
            muu = -90.0
        else:
            muu = MUSAT+KB*T*math.log(RH)
        muu_lookup.append(muu)

    _filter_wffy = tf.constant(
        [[[[0]], [[WFF * Y]], [[0]]],
         [[[WFF * Y]], [[0]], [[WFF * Y]]],
         [[[0]], [[WFF * Y]], [[0]]]],
        dtype=tf.float32
    )
    
    _filter_wff = tf.constant(
        [[[[0]], [[WFF * BETA]], [[0]]],
         [[[WFF * BETA]], [[0]], [[WFF * BETA]]],
         [[[0]], [[WFF * BETA]], [[0]]]],
        dtype=tf.float32
    )
    
    _filter_y = tf.constant(
        [[[[0]], [[Y]], [[0]]],
         [[[Y]], [[0]], [[Y]]],
         [[[0]], [[Y]], [[0]]]],
        dtype=tf.float32
    )
    
    _filter_1 = tf.constant(
        [[[[0]], [[1]], [[0]]],
         [[[1]], [[0]], [[1]]],
         [[[0]], [[1]], [[0]]]],
        dtype=tf.float32
    )
    

    r0 = tf.tile(grids, [1, 3, 3])
    r1 = tf.tile(grids, [1, 3, 3])
    rneg = 1 - r0

    r0 = r0[:, GRID_SIZE:2*GRID_SIZE, GRID_SIZE:2*GRID_SIZE]
    r1 = r1[:, GRID_SIZE-1:2*GRID_SIZE+1, GRID_SIZE-1:2*GRID_SIZE+1]
    rneg = rneg[:, GRID_SIZE-1:2*GRID_SIZE+1, GRID_SIZE-1:2*GRID_SIZE+1]

    print(r0.shape)
    print(r1.shape)
    print(GRID_SIZE)
    r0 = tf.reshape(r0, [batch_size, GRID_SIZE, GRID_SIZE, 1])
    r1 = tf.reshape(r1, [batch_size, GRID_SIZE+2, GRID_SIZE+2, 1])
    rneg = tf.reshape(rneg, [batch_size, GRID_SIZE+2, GRID_SIZE+2, 1])

    total_pores = tf.maximum(tf.reduce_sum(grids, [1, 2]), 1)

    rs = list()
    
    densities = [tf.zeros(batch_size)]
    for jj in range(1, N_ADSORP):
        for i in range(inner_loops):
            vir1 = tf.nn.conv2d(r1, strides=[1,1,1,1], filters=_filter_1, padding='VALID')
            vir0 = tf.nn.conv2d(rneg, strides=[1,1,1,1], filters=_filter_y, padding='VALID')
            vi = WFF * (vir1 + vir0) + muu_lookup[jj];

            rounew = r0 * tf.nn.sigmoid(BETA * vi)

            r1 = tf.tile(rounew, [1, 3, 3, 1])
            r1 = r1[:, GRID_SIZE-1:2*GRID_SIZE+1, GRID_SIZE-1:2*GRID_SIZE+1, :]
        rs.append(r1)
    
        density = tf.truediv(tf.reduce_sum(r1[:, 1:GRID_SIZE+1, 1:GRID_SIZE+1, :], axis=[1, 2, 3]), total_pores)
        densities.append(density)
    densities.append(tf.ones(batch_size))


    grid = grids[0]
    dft_curve = np.float32(densities)
    relative_humidity = np.arange(41) * STEP_SIZE * 100

    fig = plt.figure(figsize=(16, 8))
    fig.tight_layout(rect=[0, 0, 1, 1])
    plt.rcParams.update({'font.size': 16})

    ax = plt.subplot(212)
    ax.clear()
    ax.set_title('Adsorption Isotherm')
    ax.plot(relative_humidity, dft_curve)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Relative Humidity RH (%)')
    ax.set_ylabel("Average pore water density $\\rho$ (a.u.)")

    ax = plt.subplot(241)
    ax.clear()
    #ax.set_title('Porous Matrix (Black = Solid, White = Pore)')
    ax.set_title('Porous Matrix (RH = 0%)')
    ax.set_axis_off()
    ax.pcolor(1 - grid, cmap='Greys', vmin=0.0, vmax=1.0)
    ax.set_aspect('equal')

    water=np.array(rs)[3,0,1:-1,1:-1,0]
    ax = plt.subplot(242)
    ax.clear()
    ax.set_title('RH = 10%')
    cmap = plt.cm.Greys
    alphas = np.flip(1-grid,0)
    colors = Normalize(0, 1, clip=True)(np.flip(1-grid, 0))
    colors = cmap(colors)
    colors[..., -1] = alphas
    im = ax.imshow(np.flip(water, 0), extent=(0, GRID_SIZE, 0, GRID_SIZE), cmap='Blues', vmin=0, vmax=1)
    ax.imshow(colors, extent=(0, GRID_SIZE, 0, GRID_SIZE), cmap='Greys', vmin=0, vmax=1)
    ax.set_axis_off()
    ax.set_aspect('equal')

    water=np.array(rs)[19,0,1:-1,1:-1,0]
    ax = plt.subplot(243)
    ax.clear()
    ax.set_title('RH = 50%')
    cmap = plt.cm.Greys
    alphas = np.flip(1-grid,0)
    colors = Normalize(0, 1, clip=True)(np.flip(1-grid, 0))
    colors = cmap(colors)
    colors[..., -1] = alphas
    im = ax.imshow(np.flip(water, 0), extent=(0, GRID_SIZE, 0, GRID_SIZE), cmap='Blues', vmin=0, vmax=1)
    ax.imshow(colors, extent=(0, GRID_SIZE, 0, GRID_SIZE), cmap='Greys', vmin=0, vmax=1)
    ax.set_axis_off()
    ax.set_aspect('equal')

    water=np.array(rs)[35,0,1:-1,1:-1,0]
    ax = plt.subplot(244)
    ax.clear()
    ax.set_title('RH = 90%')
    cmap = plt.cm.Greys
    alphas = np.flip(1-grid,0)
    colors = Normalize(0, 1, clip=True)(np.flip(1-grid, 0))
    colors = cmap(colors)
    colors[..., -1] = alphas
    im = ax.imshow(np.flip(water, 0), extent=(0, GRID_SIZE, 0, GRID_SIZE), cmap='Blues', vmin=0, vmax=1)
    ax.imshow(colors, extent=(0, GRID_SIZE, 0, GRID_SIZE), cmap='Greys', vmin=0, vmax=1)
    ax.set_axis_off()
    ax.set_aspect('equal')

    plt.savefig("sorption_isotherm.jpg")

    return "done"

M = np.float32(np.loadtxt(open('%s.csv'%'AI', "rb"), delimiter=",")[np.newaxis,...])
T = 300
LDFT2D_MT(M, T)
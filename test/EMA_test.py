from SAGAN import SAGAN
from gan_modules import EMA 
# %%
sagan = SAGAN(
    n_dims=128,
    n_dis=2,
    max_resolutions=128,
    g_lr = 1e-5,#now 1e-6,
    d_lr = 3e-5,#now 1.5e-6
    g_betas=(0,0.999),
    d_betas=(0,0.999),
    initial_layer="linear",
    upsampling_mode="pooling",
    downsampling_mode="pooling",
    loss_fn="hinge",
    is_da = False
    )
mvag_sagan = SAGAN(
    n_dims=128,
    n_dis=2,
    max_resolutions=128,
    g_lr = 1e-5,#now 1e-6,
    d_lr = 3e-5,#now 1.5e-6
    g_betas=(0,0.999),
    d_betas=(0,0.999),
    initial_layer="linear",
    upsampling_mode="pooling",
    downsampling_mode="pooling",
    loss_fn="hinge",
    is_da = False
    )
ema = EMA()
ema.setup(mvag_sagan.netG)
# %%
print(next(mvag_sagan.netG.parameters()))
#%%
for i in range(300):
    ema.apply(sagan.netG,mvag_sagan.netG)

#%%
print(next(mvag_sagan.netG.parameters()))

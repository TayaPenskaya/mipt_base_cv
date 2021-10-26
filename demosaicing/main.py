from src import utils as ut
from src import vng_algo as vng
from src import metrics as m

image = ut.get_image_from_bmp('./data/RGB_CFA.bmp')
# new_image = vng.demosaicing_image(image)
# new_im = ut.get_image_from_array(new_image)
# new_im.save("received_img.jpeg")
new_image = ut.get_image_from_bmp('./data/received_img.jpeg')
real_image = ut.get_image_from_bmp('./data/Original.bmp')

print('PSNR: ', m.psnr(real_image, new_image))

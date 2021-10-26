# Demosaicing

## Solution

Demosaicing was realized by the Variable Number of Gradients algorithm.

PSNR: 16.389852617380292
Speed: 2871.66it/s (8656848 -- 50:03)

- [received image](./data/received_img.jpeg)

## Image Sensors

### CCD (Charge-coupled device)

register photon rays in silicon which contains a grid array of pixels -> electorn charges are captured in this pixel array ->
processed from the bottom to the top of the grid into serial shift register -> pushed out a single charge at a time to be converted into an analog voltage -> transformed into coding

### CMOS 

instead of shuffling electron charges along an array to then be modified -> extra circuitry has been added to each pixel which allows it to do all processing individually with the signal then being sent directly down the line to the CPU

## Useful links

- [Image sensors](https://youtu.be/2ZXamWYdUgQ)
- [Bayer filter](https://en.wikipedia.org/wiki/Bayer_filter)

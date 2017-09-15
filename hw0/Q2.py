from PIL import Image
import sys

if __name__ == "__main__":
	url = sys.argv[1]
	img1 = Image.open(url)
	width,height = img1.size
	img2 = Image.new("RGB",(width,height))

	for x in range(width):
		for y in range(height):
			r,g,b = img1.getpixel((x,y))
			r /= 2
			g /= 2
			b /= 2
			img2.putpixel((x,y),(int(r),int(g),int(b)))
	img2.save("Q2.png")
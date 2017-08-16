import json
from PIL import Image
from scipy import misc
import base64
import argparse
import sys
import cStringIO
import requests

def file2base64(imdir):
	img = Image.open(imdir)
	buffer = cStringIO.StringIO()
	img.save(buffer, format="JPEG")
	img_str = base64.b64encode(buffer.getvalue())
	return img_str

def test_recognize(args):
	imdetect = args.detect
	im1 = args.im1
	im2 = args.im2

	payload = {'img':file2base64(imdetect)}
	import numpy as np
	imarr = np.array(misc.imread(imdetect))
	r = requests.get("http://face.icybee.cn/face/face_detect", data=payload)
	print(json.loads(r.text)['boxes'][0])
	box = json.loads(r.text)['boxes'][0]
	box = [int(i) for  i in box]
	misc.imsave('sample.jpg',imarr[box[1]:box[3],box[0]:box[2],:],)

	payload = {
			'img1':file2base64(im1),
			'img2':file2base64(im2)
			}
	r = requests.get("http://face.icybee.cn/face/face_recognize", data=payload)
	print(r.text)
	#print(json.loads(r.text)['dist'])

	

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('detect',type=str,help='detect image')
	parser.add_argument('im1',type=str,help='detect image1')
	parser.add_argument('im2',type=str,help='detect image2')
	args = parser.parse_args(sys.argv[1:])
	test_recognize(args)

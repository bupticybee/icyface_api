import flask
import os
import sys
from flask import Flask
from flask import request
import json
import tensorflow as tf
import tflearn 
from align.box_seg import OpencvBoxing,CenterBlindBoxing,CascadeBoxing,ExpandMarginSegmenter,MtcnnBoxing,CascadeSegmenter
import json
import base64
import cStringIO
from PIL import Image
import numpy as np
from flask import jsonify
from scipy.misc import imresize
import argparse

app = Flask(__name__)

tf.reset_default_graph()
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
IMG_SIZE = 224
GPU_CORE = 5
HIDDEN = 160
BATCH_SIZE = 64
tflearn.config.init_training_mode()
input_layer = tflearn.input_data([None, IMG_SIZE, IMG_SIZE, 3])
net = tflearn.conv_2d(input_layer, 64, 7, strides=2)
net = tflearn.max_pool_2d(net, 3, strides=2)
net = tflearn.residual_block(net, 3, 64)
net = tflearn.residual_block(net, 1, 128, downsample=True)
net = tflearn.residual_block(net, 3, 128)
net = tflearn.residual_block(net, 1, 256, downsample=True)
net = tflearn.residual_block(net, 5, 256)
net = tflearn.residual_block(net, 1, 512, downsample=True)
net = tflearn.residual_block(net, 2, 512)
net = tflearn.global_avg_pool(net)
fully_connected = tflearn.fully_connected(net, HIDDEN, activation="relu")
result = tflearn.fully_connected(fully_connected, 7211, activation='softmax')
mom = tflearn.Momentum(0.01, lr_decay=0.1, decay_step=int(395000 / BATCH_SIZE) * 300098, staircase=True)
net = tflearn.regression(result, optimizer=mom, loss="categorical_crossentropy")
model = tflearn.DNN(net,checkpoint_path='models/model',session=sess,max_checkpoints=100,
		                            tensorboard_verbose=0)

# configure for tf session
config_box = None
# session for tensorflow boxing model
sess_box = None
# mtcnn tool for face detection
mtbox = None
# opencv tool for face detection
cvbox = None
# cascade tool for face detection
cbox = None
# expand margin segmenter ,for image clip
em = None

def preload(args):
	global model,sess,config_box,sess_box,mtbox,cvbox,cbox,em
	# load resnet for face recognization
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()
	saver.restore(sess,args.model_dir)
	# load tools
	config_box = tf.ConfigProto(allow_soft_placement=True)
	config_box.gpu_options.allow_growth = True
	sess_box = tf.Session(config=config_box)
	mtbox = MtcnnBoxing(sess=sess_box)
	cvbox = OpencvBoxing()
	cbox = CascadeBoxing(mtbox=mtbox,cvbox=cvbox)
	em = ExpandMarginSegmenter()

def recognize(picbase64):
	global sess,fully_connected,input_layer
	image = Image.open(cStringIO.StringIO(base64.b64decode(picbase64)))
	imagearr = np.asarray(image)
	imagearr = exame_image(imagearr)
	box = cbox.get_facebox(imagearr)
	segmented = em.segment(imagearr,box,None)
	segmented = segmented[0][1]
	vector = sess.run(fully_connected,feed_dict={
		input_layer:np.asarray(imresize(segmented,(224,224))
			.reshape((1,224,224,3)),dtype=np.float
			) / 255})[0]
	return box,vector

def cosin_dist(em1,em2):
	score = np.sum(em1 * em2) / np.sqrt(np.sum(np.square(em1)) * np.sum(np.square(em2)))
	return score

def np2arr(innp):
	arrshape = np.asarray(innp).shape
	if len(arrshape) == 1:
		return [float(i) for i in innp]
	elif len(arrshape) == 2:
		return [[float(j) for j in i ]for i in innp]
	else:
		raise Exception('retval dim not 2 or 3')

def exame_image(imgarr):
	print(imgarr.shape)
	assert(len(imgarr.shape) == 3)
	assert(imgarr.shape[2] == 3 or imgarr.shape[2] == 4)
	if imgarr.shape[2] == 3: # jpeg jpg image
		return imgarr
	elif imgarr.shape[2] == 4: # png image
		return imgarr[:,:,:3]

@app.route('/face/face_detect',methods=['GET', 'POST'])
def api_face_detect():
	try:
		try:
			post = request.get_json()
			imgbase64 = post.get('img')
		except:
			imgbase64 = request.form['img']
		imgdecode = base64.b64decode(imgbase64)
		cs = cStringIO.StringIO(imgdecode)
		image = Image.open(cs)
		imagearr = np.asarray(image)
		imagearr = exame_image(imagearr)
		box = cbox.get_multibox(imagearr)
		return jsonify(errcode='success',boxes=np2arr(box))
	except Exception,e: 
		import traceback 
		traceback.print_exc()  
		return jsonify(errcode='error',error=str(e))

@app.route('/face/face_recognize',methods=['GET', 'POST'])
def api_face_recognize():
	try:
		try:
			post = request.get_json()
			imgbase64_1 = post.get('img1')
			imgbase64_2 = post.get('img2')
		except:
			imgbase64_1 = request.form['img1']
			imgbase64_2 = request.form['img2']
		# process the first pic
		box1,vec1 = recognize(imgbase64_1)
		box2,vec2 = recognize(imgbase64_2)
		dist = cosin_dist(vec1,vec2)
		return jsonify(errcode='success',
				box1=np2arr(box1),
				box2=np2arr(box2),
				vec1=vec1.tolist(),
				vec2=vec2.tolist(),
				dist=float(dist))
	except Exception,e: 
		import traceback 
		traceback.print_exc()  
		return jsonify(errcode='error',error=str(e))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('model_dir',type=str,help='dir of the model')
	args = parser.parse_args(sys.argv[1:])
	print('preloading...')
	preload(args)
	app.run(port=8840,threaded=True)

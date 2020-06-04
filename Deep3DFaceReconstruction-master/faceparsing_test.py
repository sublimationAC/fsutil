import tensorflow as tf 
import numpy as np
import cv2
from PIL import Image
import os
import glob
from scipy.io import loadmat,savemat

from preprocess_img import Preprocess
from load_data import *
from reconstruct_mesh import Reconstruction
# from mesh_render import render
from mesh_numpy import render as renderp
from mesh_render import render 
import reduce_ill
import pre_data_util as pr_ut

def load_graph(graph_filename):
	with tf.gfile.GFile(graph_filename,'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())

	return graph_def


n_ver=35709
number_lables=17
result_cnt=np.zeros((n_ver,number_lables),dtype=np.int)
idx_0ton=list(range(n_ver))
n_ld=68
result_cnt_ld=np.zeros((n_ld,number_lables),dtype=np.int)
idx_0ton_ld=list(range(n_ld))
def cnt_idx_label(input_label,image_vertices,ldmk_vertices,norm):
	image_vertices = np.squeeze(image_vertices)
	image_vertices[:,2]*=-1
	norm=np.squeeze(norm)
	# norm[:,2]*=-1
	input_label=np.squeeze(input_label)
	# print('input_label',input_label.shape)
	# (Image.fromarray(np.around(input_label).astype(np.uint8))).save('debug.jpg')
	# cv2.imshow('input_label',input_label*10)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	image_vertices_xy=np.around(image_vertices[:,:2]).astype(np.int)	
	p=input_label[list(image_vertices_xy[:,1]),list(image_vertices_xy[:,0])]
	
	# print('p shape',p.shape,'norm shape:',(norm[:,2]>0).shape,'idx_0ton sh',len(idx_0ton))
	result_cnt[idx_0ton,list(p)]+=norm[:,2]>0
	print('norm[:,2]>0',np.sum(norm[:,2]>0))
	ldmk_vertices=np.around(ldmk_vertices).astype(np.int)	
	result_cnt_ld[idx_0ton_ld,list(input_label[list(ldmk_vertices[:,1]),list(ldmk_vertices[:,0])])]+=1



def demo():
	# input and output folder
	# image_path = 'test_one'
	root_path = '/data/weiliu/faceparsing/train'		
	ldmk_list = glob.glob(root_path+'_aligned/bbxNldmk/' + '/**/' + '*.mat', recursive=True)
	# print(root_path+'_aligned/bbxNldmk/' + '/*/' + '*.mat')
	# print('ldmk_list',ldmk_list)
	# read BFM face model
	# transfer original BFM model to our model
	if not os.path.isfile('./BFM/BFM_model_front.mat'):
		transferBFM09()
	# transferBFM09()

	# read face model
	facemodel = BFM()
	# read standard landmarks for preprocessing images
	lm3D = load_lm3d()
	n = 0
	
	# build reconstruction model
	with tf.Graph().as_default() as graph,tf.device('/cpu:0'):

		images = tf.placeholder(name = 'input_imgs', shape = [None,224,224,3], dtype = tf.float32)
		graph_def = load_graph('network/FaceReconModel.pb')
		tf.import_graph_def(graph_def,name='resnet',input_map={'input_imgs:0': images})

		# output coefficients of R-Net (dim = 257) 
		coeff = graph.get_tensor_by_name('resnet/coeff:0')

		with tf.Session() as sess:
			print('reconstructing...')
			for ldmkfile in ldmk_list:
				n += 1
				print(n)
				# print('ldmkfile:',ldmkfile)
				# load images and corresponding 5 facial landmarks
				img_path_name=ldmkfile.replace('mat','jpg').replace('_aligned/bbxNldmk','/jpg')
				lable_path_name=ldmkfile.replace('mat','png').replace('_aligned/bbxNldmk','/mask')				

				img=Image.open(img_path_name)
				label=Image.open(lable_path_name)
				lm=loadmat(ldmkfile)['facial5points']
				
				# preprocess input image
				input_img,lm_new,transform_params = Preprocess(img,lm,lm3D)
				input_label,_,__ = Preprocess(label,lm,lm3D)

				coef = sess.run(coeff,feed_dict = {images: input_img})

				# reconstruct 3D face with output coefficients and face model
				face_shape,face_texture,face_color,tri,face_projection,z_buffer,landmarks_2d,light,norm = Reconstruction(coef,facemodel)
				# print('norm sp demo',norm.shape)
				input_img = np.squeeze(input_img)
				# cv2.imshow('input_img',input_img)
				# cv2.waitKey(0)
				# cv2.destroyAllWindows()
				# reshape outputs
				shape = np.squeeze(face_shape, (0))
				color = np.squeeze(face_color, (0))
				landmarks_2d = np.squeeze(landmarks_2d, (0))


				cnt_idx_label(input_label,face_projection,landmarks_2d,norm)
				
	savemat(root_path+'/vtx_cnt.mat',{'result_cnt':result_cnt,'result_cnt_ld':result_cnt_ld})
	print(result_cnt_ld)
__define_all__=1
def get_lable_img(input_img,triangles,image_vertices,vtx_lable_color):
	input_img=input_img[:,:,::-1]
	h, w=input_img.shape[:2]
	image_vertices = np.squeeze(image_vertices)
	image_vertices[:,2]*=-1
	
	triangles=np.around(triangles).astype(np.int)
	triangles=triangles-1

	if (__define_all__==0):
		for lb in [2,3,4,5]:
			idx=np.where(vtx_lable_color=lb)
			color=np.ones(idx.shape[0],dtype=np.int)*lb
	else:
		result=render.render_colors(image_vertices, triangles, vtx_lable_color, h, w,1)
	
	return result

def get_lable_color(matpath='/data/weiliu/faceparsing/train/vtx_cnt.mat'):
	vtx_cnt=loadmat(matpath)['result_cnt']
	lable=vtx_cnt.argmax(axis=1)
	lable=lable[:,np.newaxis]
	return lable

def get_lable():

	root_path = '/data/weiliu/faceparsing/val'		
	ldmk_list = glob.glob(root_path+'_aligned/bbxNldmk/' + '/**/' + '*.mat', recursive=True)
	if not os.path.isfile('./BFM/BFM_model_front.mat'):
		transferBFM09()
	facemodel = BFM()
	lm3D = load_lm3d()

	vtx_lable_color=get_lable_color()#root_path+'/vtx_cnt.mat')
	
	n = 0
	with tf.Graph().as_default() as graph,tf.device('/cpu:0'):
		images = tf.placeholder(name = 'input_imgs', shape = [None,224,224,3], dtype = tf.float32)
		graph_def = load_graph('network/FaceReconModel.pb')
		tf.import_graph_def(graph_def,name='resnet',input_map={'input_imgs:0': images})
		coeff = graph.get_tensor_by_name('resnet/coeff:0')
		with tf.Session() as sess:
			print('reconstructing...')
			for ldmkfile in ldmk_list:
				n += 1
				print(n)

				img_path_name=ldmkfile.replace('mat','jpg').replace('_aligned/bbxNldmk','/jpg')				
				save_file_name=ldmkfile.replace('mat','png').replace('_aligned/bbxNldmk','/mask_3d')

				if (not os.path.exists(os.path.dirname(save_file_name))):
					os.makedirs(os.path.dirname(save_file_name))
			
				img=Image.open(img_path_name)				
				lm=loadmat(ldmkfile)['facial5points']
				# img.show('img')

				input_img,lm_new,transform_params = Preprocess(img,lm,lm3D)

				coef = sess.run(coeff,feed_dict = {images: input_img})

				# reconstruct 3D face with output coefficients and face model
				face_shape,face_texture,face_color,tri,face_projection,z_buffer,landmarks_2d,light,norm = Reconstruction(coef,facemodel)
				# print('norm sp demo',norm.shape)
				input_img = np.squeeze(input_img)

				shape = np.squeeze(face_shape, (0))
				color = np.squeeze(face_color, (0))
				landmarks_2d = np.squeeze(landmarks_2d, (0))
				lable_img=get_lable_img(input_img,tri,face_projection,vtx_lable_color)
				# print('lable_img max min',lable_img.max(),lable_img.min())
				# cv2.imshow('lable_img',lable_img*10/255)
				# cv2.waitKey(0)
				# cv2.destroyAllWindows()
				lable_img,lm_i=reduce_ill.inv_process_img(lable_img,lm_new,transform_params)
				# print(lm_i==lm)
				# lable_img.show('lable_img')
				# cv2.imshow('img',np.array(img))
				print('lable_img',np.array(lable_img).max(),np.array(lable_img).min())
				# cv2.imshow('lable_img',np.array(lable_img)*10)
				# cv2.waitKey(0)
				# cv2.destroyAllWindows()
				lable_img.save(save_file_name)

				
if __name__ == '__main__':
	# demo()
	get_lable()
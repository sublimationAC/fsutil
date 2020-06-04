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
from mesh_render import render
# from mesh_numpy import render

def load_graph(graph_filename):
	with tf.gfile.GFile(graph_filename,'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())

	return graph_def

import reduce_ill

def save_result(file_name,input_img,face_texture,face_color,triangles,image_vertices,light_coef,z_buffer,lm5,transform_params):

	face_texture = np.squeeze(face_texture)
	face_color = np.squeeze(face_color)
	image_vertices = np.squeeze(image_vertices)
	light_coef = np.squeeze(light_coef)
	triangles=np.around(triangles).astype(np.int)
	triangles=triangles-1
	input_img=input_img[:,:,::-1]
	# print('img texture color tri,face_projection,light shape',input_img.shape,
	#face_texture.shape,face_color.shape,triangles.shape,image_vertices.shape,light_coef.shape)
	#squeeze

	# (input_img).save(file_name[:-5]+'_in.jpg')

	h,w,c=input_img.shape
	result=np.zeros((h*2,w*4,3), dtype = np.float32)    

	result[:h,:w]=input_img
	# for i in range(5):
	#     print('i: ',i*30,'img:',input_img[i*30,i*30],'result:',result[i*30,i*30])


	# print("face_color max",face_color.max())
	rendering = render.render_colors(image_vertices, triangles, face_color, h, w,3,input_img)
	rendering = np.minimum((np.maximum(rendering, 0)), 255)
	# face_color_ren=rendering.copy()
	result[:h,w:2*w]=rendering

	# print("face_texture max",face_texture.max())
	rendering = render.render_colors(image_vertices, triangles, face_texture, h, w)
	rendering = np.minimum((np.maximum(rendering, 0)), 255)        
	# face_texture_ren=rendering.copy()
	result[:h,2*w:3*w]=rendering


	# normal=normal+np.array([1,1,1])
	# # print('normal shape',normal.shape)
	# mag = np.sum(normal**2, 1) # [nver]
	# # print('mag shape',mag.shape)
	# normal = normal/np.sqrt(mag[:,np.newaxis])
	# result[h:,:w]=mesh.render.render_colors(image_vertices, triangles, normal, h, w)

	# light_coef=light_coef.sum(axis=0)
	# print('light coef',light_coef.shape)
	rendering = render.render_colors(image_vertices, triangles, light_coef, h, w)
	rendering = np.minimum((np.maximum(rendering, 0)), 255)
	lmy=lm5.copy()
	ill_coef_ren=rendering.copy()
	ill_coef_p=reduce_ill.inv_process_img(rendering,lm5,transform_params)
	result[h:,w:2*w]=rendering

	light_coef=light_coef/128
	zero_ind = (light_coef == 0)
	# # print('light_coef zero_ind shape',zero_ind.shape)
	light_coef[zero_ind] = 1
	# # print('lit_colors light_coef shape',face_color.shape,light_coef.shape)
	div_l=face_color/light_coef
	# div_l=face_texture*light_coef
	rendering = render.render_colors(image_vertices, triangles, div_l, h, w)
	rendering = np.minimum((np.maximum(rendering, 0)), 255)
	result[h:,2*w:3*w]=rendering



	rendering = reduce_ill.reduce_img_ill(input_img,image_vertices, triangles, light_coef, h, w)
	rendering = np.minimum((np.maximum(rendering, 0)), 255)
	result[:h,3*w:]=rendering
	ini_rd_il=reduce_ill.inv_process_img(rendering,lmy,transform_params)
	# result[h:,:w]=rendering

	rco=(ill_coef_ren/128.)*rendering
	rco = np.minimum((np.maximum(rco, 0)), 255)
	result[h:,3*w:]=rco
	# for x in range(1,10):
	# 	print(x*10)
	# 	t=x*10
	# 	print('input_img ',input_img[t][t] ,' rco',rco[t][t])
	# 	print('ini_rd_il[0]',rendering[t][t],'ill_coef_ren',ill_coef_ren[t][t]/128.)
	# 	print('face_color_ren ',face_color_ren[t][t],'ill_coef_ren',face_texture_ren[t][t])
	rco=reduce_ill.inv_process_img(rco,lmy,transform_params)
	rco[0].save(file_name[:-4]+'_rco_itf_il.jpg')

	print('name:',file_name[:-4]+'_out.jpg')
	(Image.fromarray(np.around(result).astype(np.uint8))).save(file_name[:-4]+'_out.jpg')

	ini_rd_il[0].save(file_name[:-4]+'ini_rd_ill.jpg')
	ill_coef_p[0].save(file_name[:-4]+'_ill.jpg')

	rco=np.array(ini_rd_il[0])*(np.array(ill_coef_p[0])/128.)
	rco = np.minimum((np.maximum(rco, 0)), 255)		
	(Image.fromarray(np.around(rco).astype(np.uint8))).save(file_name[:-4]+'_rco_il.jpg')

	# print('ini_rd_il ',ini_rd_il[1])
	# print('ill_coef_p ',ill_coef_p[1])


def demo():
	# input and output folder
	image_path = 'fs_p'#'test_1'#'illumination'#'input'
	save_path = 'output'	
	img_list = glob.glob(image_path + '/' + '*.png')+glob.glob(image_path + '/' + '*.jpg')

	# read BFM face model
	# transfer original BFM model to our model
	if not os.path.isfile('./BFM/BFM_model_front.mat'):
		transferBFM09()

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
			for file in img_list:
				n += 1
				print(n)
				# load images and corresponding 5 facial landmarks
				print(file)
				
				# img,lm = load_img(file,file.replace('png','txt'))
				img,lm = load_img(file,file[:-3]+'txt')
				img=img.convert("RGB")
                # img.save(save_path+'/'+file[:-4]+'ini.jpg')
				# preprocess input image
				lm_ini=lm.copy()
				input_img,lm_new,transform_params = Preprocess(img,lm,lm3D)

				coef = sess.run(coeff,feed_dict = {images: input_img})

				# reconstruct 3D face with output coefficients and face model
				face_shape,face_texture,face_color,tri,face_projection,z_buffer,landmarks_2d,light = Reconstruction(coef,facemodel)

				# reshape outputs
				input_img = np.squeeze(input_img)
				shape = np.squeeze(face_shape, (0))
				color = np.squeeze(face_color, (0))
				landmarks_2d = np.squeeze(landmarks_2d, (0))
				save_result(save_path+'/'+file,input_img,face_texture,face_color,tri,face_projection,light,z_buffer,lm_new,transform_params)
				# print(lm_ini,lm)
				print(lm_ini==lm)
				# save output files
				# cropped image, which is the direct input to our R-Net
				# 257 dim output coefficients by R-Net
				# # 68 face landmarks of cropped image
				# savemat(os.path.join(save_path,file.split(os.path.sep)[-1].replace('.png','.mat')),{'cropped_img':input_img[:,:,::-1],'coeff':coef,'landmarks_2d':landmarks_2d,'lm_5p':lm_new})
				# save_obj(os.path.join(save_path,file.split(os.path.sep)[-1].replace('.png','_mesh.obj')),shape,tri,np.clip(color,0,255)/255) # 3D reconstruction face (in canonical view)

if __name__ == '__main__':
	demo()

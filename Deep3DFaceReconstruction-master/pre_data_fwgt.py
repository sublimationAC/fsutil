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

def save_result(file_name,input_img,face_texture,face_color,triangles,
	image_vertices,light_coef,z_buffer,lm5,transform_params,uv_coords,norm,coef):

	face_texture = np.squeeze(face_texture)
	face_color = np.squeeze(face_color)
	image_vertices = np.squeeze(image_vertices)
	image_vertices[:,2]*=-1
	light_coef = np.squeeze(light_coef)
	norm=np.squeeze(norm)
	# norm[:,2]*=-1
	triangles=np.around(triangles).astype(np.int)
	triangles=triangles-1
	input_img=input_img[:,:,::-1]
	# print('img texture color tri,face_projection,light shape',input_img.shape,
	#face_texture.shape,face_color.shape,triangles.shape,image_vertices.shape,light_coef.shape)
	#squeeze

	# (input_img).save(file_name[:-5]+'_in.jpg')

	h,w,c=input_img.shape
	result=np.zeros((h*2,w*4,3), dtype = np.float32)    

	# for i in range(5):
	#     print('i: ',i*30,'img:',input_img[i*30,i*30],'result:',result[i*30,i*30])

	
	result[:h,:w]=input_img
	# print('input_img:','ma: ',input_img.max(),' mi :', input_img.min())
	# print('norm',norm, ' norm z>0',np.sum(norm[:,2]>0))	
	rendering = render.render_colors(image_vertices, triangles, norm, h, w,3)
	norm_mat=rendering.copy()
	rendering=rendering+np.array([1,1,1])
	rendering = rendering/np.expand_dims(np.linalg.norm(rendering,axis = 2),2)*255
	rendering = np.minimum((np.maximum(rendering, 0)), 255)		
	# rendering[:,:,:2]=0
	result[:h,w:2*w]=rendering	
	One255=np.array([[255,255,255]],dtype=np.float32)
	# print('np.repeat(One255,norm.shape[0],axis=0)',np.repeat(One255,norm.shape[0],axis=0).shape)
	rendering = render.render_colors(image_vertices, triangles, np.repeat(One255,norm.shape[0],axis=0), h, w,3)
	rendering = np.minimum((np.maximum(rendering, 0)), 255)
	result[:h,2*w:3*w]=rendering


	vis_pts=renderp.get_block_idx(image_vertices, triangles, norm, h, w,3)

	print('vis pts render vs norm',np.sum(vis_pts==(norm[:,2]>0)),vis_pts.shape)
	print('vis pts render vs norm',(vis_pts==(norm[:,2]>0)).all())
	image_vertices_xy=np.around(image_vertices[:,:2]).astype(np.int)
	color_vis_init=input_img[list(image_vertices_xy[:,1]),list(image_vertices_xy[:,0])]*vis_pts[:,np.newaxis]
	# print('uv_coords shape: ',uv_coords.shape)
	rendering = render.render_colors(uv_coords, triangles, color_vis_init, h, w,3)
	rendering = np.minimum((np.maximum(rendering, 0)), 255)
	result[h:,:w]=rendering
	# print("face_texture max",face_texture.max())
	rendering = render.render_colors(uv_coords, triangles, norm, h, w, 3)
	norm_UV_mat=rendering.copy()
	rendering=rendering+np.array([1,1,1])
	rendering = rendering/np.expand_dims(np.linalg.norm(rendering,axis = 2),2)*255
	rendering = np.minimum((np.maximum(rendering, 0)), 255)        
	result[h:,w:2*w]=rendering

	rendering = render.render_colors(uv_coords, triangles, 
		np.repeat(One255,norm.shape[0],axis=0)*vis_pts[:,np.newaxis], h, w,3)
	rendering = np.minimum((np.maximum(rendering, 0)), 255)
	result[h:,2*w:3*w]=rendering

	uv_coords[:,2]=image_vertices[:,2]
	UVfromIMG_mat=render.render_colors(uv_coords, triangles, image_vertices, h, w,3)
	IMGfromUV_mat=render.render_colors(image_vertices, triangles, uv_coords, h, w,3)
	rendering=UVfromIMG_mat.copy()
	rendering=rendering+np.array([0,0,1])
	rendering = rendering/np.expand_dims(np.linalg.norm(rendering,axis = 2),2)*255
	rendering = np.minimum((np.maximum(rendering, 0)), 255)        
	result[h:,3*w:4*w]=rendering
	rendering=IMGfromUV_mat.copy()
	rendering=rendering+np.array([0,0,1])
	rendering = rendering/np.expand_dims(np.linalg.norm(rendering,axis = 2),2)*255
	rendering = np.minimum((np.maximum(rendering, 0)), 255)  
	result[:h,3*w:4*w]=rendering

	print('name:',file_name.replace('.land73','_fit.jpg'))
	(Image.fromarray(np.around(result).astype(np.uint8))).save(file_name.replace('.land73','_fit.jpg'))
	
	savemat(os.path.join(file_name.replace('.land73','.mat'))
	,{'cropped_img':input_img[:,:,::-1],'norm_mat':norm_mat,'norm_UV_mat':norm_UV_mat,'UVfromIMG':UVfromIMG_mat,'IMGfromUV':IMGfromUV_mat,
	'coeff':coef,'visible_pts':vis_pts})


def demo():
	# input and output folder
	# image_path = 'test_one'
	image_path = 'fwgt'
	save_path = 'fit_fwgt'	
	img_list = glob.glob(image_path + '/**/' + '*.png', recursive=True) \
				+glob.glob(image_path + '/**/' + '*.jpg', recursive=True) \
				+glob.glob(image_path + '/**/' + '*.bmp', recursive=True)

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

	uv_coords = pr_ut.load_uv_coords('BFM/BFM_UV.mat') 
	uv_h = uv_w = 224	
	uv_coords = pr_ut.process_uv(uv_coords, uv_h, uv_w)
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
				path_name=file.replace('png','land73').replace('jpg','land73').replace('bmp','land73')
				if (not os.path.exists(path_name)):
					continue
				if (not os.path.exists(save_path+'/'+os.path.dirname(path_name))):
					os.makedirs(save_path+'/'+os.path.dirname(path_name))
			
				img,lm = pr_ut.load_img_73(file,path_name)
				# preprocess input image
				input_img,lm_new,transform_params = Preprocess(img,lm,lm3D)

				coef = sess.run(coeff,feed_dict = {images: input_img})

				# reconstruct 3D face with output coefficients and face model
				face_shape,face_texture,face_color,tri,face_projection,z_buffer,landmarks_2d,light,norm = Reconstruction(coef,facemodel)

				# reshape outputs
				input_img = np.squeeze(input_img)
				shape = np.squeeze(face_shape, (0))
				color = np.squeeze(face_color, (0))
				landmarks_2d = np.squeeze(landmarks_2d, (0))

				# save output files
				# cropped image, which is the direct input to our R-Net
				# 257 dim output coefficients by R-Net
				# 68 face landmarks of cropped image
				save_result(save_path+'/'+path_name,input_img,face_texture,face_color,tri,
				face_projection,light,z_buffer,lm_new,transform_params,uv_coords,norm,coef)

				# savemat(os.path.join(save_path,file.split(os.path.sep)[-1].replace('.png','.mat').replace('.jpg','.mat'))
				# ,{'cropped_img':input_img[:,:,::-1],'coeff':coef,'landmarks_2d':landmarks_2d,'lm_5p':lm_new})
				save_obj(os.path.join(save_path,path_name.replace('.land73','_mesh.obj'))
				,shape,tri,np.clip(color,0,255)/255) # 3D reconstruction face (in canonical view)

if __name__ == '__main__':
	demo()

##Physics based scene generation
**/mnt/disk/scenenet/chrono/src/demos/irrlicht** is the directory that contains the main file **demo_collision_trimesh.cpp** that generates the scenes, it contains 

- scaling and sampling the objects from ShapeNets
- picking the room layout 
- running the physics 
- saving a .txt file with object wnids, object poses (transformations), layout fileName.
- scale is the height in meters *i.e.* $$ scale \times \frac{y}{max_y - min_y}$$
- mapping from the object name to the corresponding texture from the texture_library was done by hand.

```
layout_file: ./kitchen/kitchen8_layout.obj
object
03001627/5539a4a9f2f1b45f3eec26c23f5bc80b
wnid
04331277
scale
0.803991
transformation
0.93618 0.00954775 -0.351392 -2.20642
-0.0109176 0.999939 -0.00191719 -0.0741752
0.351352 0.0056312 0.936226 -1.12748

object
04379243/8f29431ef2b28d27bfb1fc5d146cf068
wnid
04379243
scale
0.315817
transformation
0.985232 -0.0818803 -0.150381 -1.6889
0.0808304 0.996642 -0.0130912 2.59301
0.150948 0.000742512 0.988541 1.38842

```
## ScenenetLayouts directory
- The following code in the file **changematname_importOBJ.py** ensures that the object name and the material name are the same.
```
import bpy, bmesh
import sys

print ("First argument: %s" % str(sys.argv[5]))
full_path_to_file = sys.argv[5] # + '.obj' #"night_stand_0026.obj"


print('sys.argv[0] =', sys.argv[0])
print('sys.argv[1] =', sys.argv[1])
print('sys.argv[2] =', sys.argv[2])
print('sys.argv[3] =', sys.argv[3])
print('sys.argv[4] =', sys.argv[4])
print('sys.argv[5] =', sys.argv[5])

bpy.ops.import_scene.obj(filepath=full_path_to_file, use_split_groups=True, use_split_objects=True, use_image_search=True, use_smooth_groups=False, use_groups_as_vgroups=False)


C = bpy.context
D = bpy.data

objects = [obj for obj in D.objects if obj.type == 'MESH']

for obj in objects:
    objname=obj.name
    mat=D.materials.new(objname)
    obj.data.materials.clear()
    obj.data.materials.append(mat)

#argv = sys.argv
#argv = argv[argv.index("--") + 1:] # get all args after "--"

obj_out = './' + full_path_to_file

bpy.ops.export_scene.obj(filepath=obj_out, axis_forward='-Z', axis_up='Y',
                        use_normals=True, use_uvs=True, use_materials=True)
```
- The file **scenenet_mat_to_wnid.txt** contains the mapping from the scenenet object name to their corresponding wnid. It looks like this
```
lightbulb 03665924
bulb 03665924
shield 04192858
chandelier 03005285
terrace 03899768
pane 03881893
railing 04047401
dish 03206908
carpet 04118021
....
```

- train_layouts.txt, test_layouts.txt and val_layouts.txt contain the corresponding train/test/val layouts used when creating scenes.
- The file **blender_import_export_with_UVs_allobjects.py** contains the python code to do UVmap the objects using the cube-mapping option in the blender.

## ShapeNetsObj3 directory
- We ran deinterlace_png_list.sh to fix the interlacing issues CVD had while reading a png file. 
- filtered_model_info_and_texture.txt has information about various different attributes of the 3D models *e.g.* some models do not have any texture so we don't want to render them and this is the root of all the model information we need and has space separated columns with names 
	- directory of the 3D model 
	- wordnet Id
	- tags 
	- up vector
	- front vector 
	- textured (we ignore it!) 
	- valid_texture (so is this!)
- Just to double check - there are other .txt files like **ordered_filtered_model_info_and_texture.txt** which we don't use but it was just there as a copy of some form of sorting we did.
- Additionally there are three more files with the following names 
 	- test_split_filtered_model_info_and_texture.txt
 	- train_split_filtered_model_info_and_texture.txt
 	- valid_split_filtered_model_info_and_texture.txt
 	- This is just to ensure that the train/test/valid are fully separated such that no overlapping objects among these.
- We manually removed certain models from the ShapeNets that were deemed too unrealistic as a result of their skewed aspect ratio. We also removed objects like guns, cars, planes etc.. The following codes filteres them out.

```
with open('model_info_and_texture.txt') as f:
    model_lines = f.readlines()

with open('./rejected_models_for_aspect.txt') as f:
    rejected_models = f.readlines()

reject_dict = {}
for rejected_model in rejected_models:
    reject_dict[rejected_model.rstrip()] = True

print len(reject_dict)
print len(model_lines) - 1

f = open('filtered_model_info_and_texture.txt','w')

f.write(model_lines[0])
number_written = 0
for line in model_lines[1:]:
    model = line.split()[0]
    if model.rstrip() in reject_dict:
        continue
    f.write(line)
    number_written += 1

f.close()
```
- We have this **rejected_models_for_aspect.txt** file that stores the model paths of those 3D models which have skewed aspect ratio. So how did we figure out that the aspect ratio is skewed? We divides max_y with max_z and/or max_y with max_x 
 
## Trajectory Generation code

The directory name is **scenenetplayground** which contains the main file named **main_check_room_camera_intersection.cpp**. What does it need? 

- it has **std::string layout_fileName = std::string(argv[1]);**  where argv[1] is the txt output from the physics engine.
- There is the .sh file **run_many.py** in the build of scenenetplayground. It reads the files from train_physics_layouts directory and then runs the ./room_camera_intersection and once it has done processing this file, it writes the file into processed directory and at the same time it writes the corresponding layout-object-trajectory file in **/mnt/disk/scenenet/ScenenetLayouts/train_text_layouts/**. When doing batch rendering, the OptiX code reads the files from this directory and once it has done rendering, the python script moves the file into the _processed version. 

```
from functools import partial
from multiprocessing.dummy import Pool
from subprocess import call
import os
import time
import shutil

def main():

    def get_command(file):
        filename = os.path.split(file)[1].split('.')[0]
        return './room_camera_intersection {0} {1} > /dev/null 2>&1'.format(file,filename)

    input_dir = '/mnt/disk/scenenet/chrono/train_physics_layouts/'
    output_dir = '/mnt/disk/scenenet/chrono/train_physics_layouts_processed/'

    pool = Pool(10)
    while True:
        all_files = []
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                abs_path = os.path.join(root,file)
                all_files.append(abs_path)
        # This slight gap ensures that if a file was just being written to, it
        # will have completed by the time we process it.  Writing the file takes
        # less than 10ms normally.
        time.sleep(5)
        commands = [get_command(x) for x in all_files]
        try:
            for i, returncode in enumerate(pool.imap(partial(call, shell=True), commands)):
                if returncode == 0:
                    print('Completed {0}'.format(all_files[i]))
                    try:
                        shutil.move(all_files[i],output_dir)
                    except:
                        pass
        except:
            pass

if __name__ == '__main__':
    main()

```
There is a corresponding **run_make.py** in the directory **/mnt/disk/scenenet/bin2/build** that renders scene and puts the processed files in the **/mnt/disk/scenenet/ScenenetLayouts/train_text_layouts_processed/** directory.
```
import logging
import multiprocessing
import os
import shutil
import subprocess
import threading
import time

class ThreadWorker(threading.Thread):
    def __init__(self, in_queue, queue_func, **kwargs):
        threading.Thread.__init__(self)
        self.in_queue = in_queue
        self.queue_func = queue_func
        for key, value in kwargs.items():
            setattr(self, key, value)

    def run(self):
        while True:
            try:
                item = self.in_queue.get()
            except OSError as e:
                logging.info('Queue closed')
                break
            logging.info('Queue size:{0}'.format(self.in_queue.qsize()))
            self.queue_func(self,item)
            self.in_queue.task_done()


def main():
    # Logging is thread safe
    logging.basicConfig(level=logging.INFO, format='%(asctime)s (%(threadName)-2s) %(message)s',)

    gpu_list = [0,2,3]
    input_dir = '/mnt/disk/scenenet/ScenenetLayouts/train_text_layouts/'
    processed_dir = '/mnt/disk/scenenet/ScenenetLayouts/train_text_layouts_processed/'
    output_dir = '/mnt/disk/scenenet/newdataset/'
    time_limit = 3 * 60 * 60

    # The main worker function
    def my_queue_func(worker,abs_path):
        gpu_id = worker.gpu_id
        filename_without_suffix = os.path.split(abs_path)[1].split('.')[0]
        this_run_base_dir = os.path.join(output_dir,filename_without_suffix)
        logging.info('About to process:{0} on gpu:{1}'.format(abs_path,gpu_id))
        if not os.path.exists(this_run_base_dir):
            os.mkdir(this_run_base_dir)
            os.mkdir(os.path.join(this_run_base_dir,'instance'))
            os.mkdir(os.path.join(this_run_base_dir,'photo'))
            os.mkdir(os.path.join(this_run_base_dir,'depth'))
        command = 'export CUDA_VISIBLE_DEVICES={0} && ./Headless_SceneNet {1} {2} > {1}/output.log 2>&1'.format(gpu_id,this_run_base_dir,abs_path)
        try:
            result = subprocess.call(command,shell=True,timeout=time_limit)
            if result == 0:
                logging.info('Success: finished processing:{0} return code:{1}'.format(abs_path,result))
                shutil.move(abs_path,processed_dir)
            else:
                logging.info('Failure: finished processing:{0} return code:{1}'.format(abs_path,result))
        except subprocess.TimeoutExpired as e:
            logging.info('Timeout on processing:{0}'.format(abs_path))

    while True:
        in_queue = multiprocessing.JoinableQueue()
        # Get the list of files
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                abs_path = os.path.join(root,file)
                in_queue.put(abs_path)
        time.sleep(5)
        # Spawn a pool of threads, and pass them the queue instance 
        for gpu in gpu_list:
            worker = ThreadWorker(in_queue,my_queue_func,gpu_id=gpu)
            worker.setDaemon(True)
            worker.start()
        in_queue.join()

if __name__ == '__main__':
    main()
    
```

##OptiX rendering

## How to get the word-net ids. We created a file with 155 objects and their definitions.
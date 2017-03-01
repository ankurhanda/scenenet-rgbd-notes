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
- The details about mapping Scenenet object with SUN RGB-D is here in the /mnt/disk/scenenet/chrono/canon_textfiles

```
[ankur@bigboy canon_textfiles]$ ls
class_sampling_prob.py  objects_in_scene.txt  ShapeNetsSUNRGBDMapping.txt  uniq_object_wnids.txt      val_layouts.txt
get_uniq.py             rejected_models.txt   test_layouts.txt             unique_scenes.txt          wnid_sample_probabilities_english.txt
layouts.txt             scene_type.txt        train_layouts.txt            uniq_wnids_in_dataset.txt  wnid_sample_probabilities.txt
[ankur@bigboy canon_textfiles]$ 
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
- /mnt/disk/scenenet/bin2/src/headless_SceneNet.cpp is the main file we used to render the scenes.
- run_main.py is the file we use to invoke the rendering with the OptiX render.
- This is the main function in the headless_SceneNet.cpp 
```
int main(int argc, char* argv[])
{
    std::string base_obj_folder = "/mnt/disk/scenenet/ShapeNetObj3/";
    std::string base_layout_folder = "/mnt/disk/scenenet/ScenenetLayouts/";

    //Set save base location
    std::string save_base = std::string(argv[1]);
    std::string layout_file = std::string(argv[2]);
    std::cout<<"Layout text:"<<layout_file<<std::endl;

    BaseScene scene;

    scene.initScene(save_base,layout_file,base_obj_folder,base_layout_folder,0);

    const int number_trajectory_steps = 100000;

    for (int i = 0; i < number_trajectory_steps; ++i)
    {
        if (!scene.trace(save_base+"/",i))
        {
          std::cout<<"Breaking"<<std::endl;
          break;
        }
    }
    return 0;
}
```
- /mnt/disk/scenenet/bin2/src/Scene.h and /mnt/disk/scenenet/bin2/src/Scene.cpp are where the scene is defined. 
- /mnt/disk/scenenet/bin2/src/Geometry/TriangleMesh.cu - this is just ensure that duplicate faces are rendered properly. 
	- Gets the normal that is pointing towards the ray i.e. opposite to ray direction. If it is not then flip it such that it is always pointing opposite to the ray direction. More importantly, we ignore the normals given by the .obj and use geometric normals instead. Why did we do that? Normals given by obj were not necessarily credible mostly because the models had bad normal meta data. 
	- Calling rtPotentialIntersection returns true if the t value given could potentially be an intersection point.  If there is no texture we increase the t value by 0.001, this is designed to give priority to faces that have texture.  Because they will have a smaller t value meaning they intersected first and so will be the ones returned.
- We can change the settings for rendering in the OptixRenderer.cpp file in the /mnt/disk/scenenet/bin2/src/Renderer

```
const unsigned int OptixRenderer::PHOTON_GRID_MAX_SIZE = 0;
const unsigned int OptixRenderer::MAX_PHOTON_COUNT = MAX_PHOTONS_DEPOSITS_PER_EMITTED;
const unsigned int OptixRenderer::PHOTON_LAUNCH_WIDTH = 512;
const unsigned int OptixRenderer::PHOTON_LAUNCH_HEIGHT = 1024;
// Ensure that NUM PHOTONS are a power of 2 for stochastic hash
const unsigned int OptixRenderer::EMITTED_PHOTONS_PER_ITERATION = OptixRenderer::PHOTON_LAUNCH_WIDTH*OptixRenderer::PHOTON_LAUNCH_HEIGHT;

const unsigned int OptixRenderer::NUM_PHOTON_ITERATIONS = 32;
const unsigned int OptixRenderer::NUM_PHOTONS = OptixRenderer::EMITTED_PHOTONS_PER_ITERATION*OptixRenderer::NUM_PHOTON_ITERATIONS*OptixRenderer::MAX_PHOTON_COUNT;

const unsigned int OptixRenderer::NUM_PHOTON_MAPS = 4;
const unsigned int OptixRenderer::RES_DOWNSAMPLE = 1;
const unsigned int OptixRenderer::NUM_ITERATIONS = 6;
```
- If there is some runtime error to do with optix rendering (or memory issues), remember to fix it by changing GPU settings in CMakeLists.txt file.
	
## How to get the word-net ids. We created a file with 155 objects and their definitions.
```
import math
import re
import os
import sys
import numpy as np
import PIL.Image
import numpy
from collections import namedtuple
from nltk.corpus import wordnet as wn

# The additional categories lookup to a string only - the official wordnet objects return a 
# synset object with definitions, and an appropriate place in the graph
def get_id_to_wnsynset_dict():
    additional_cats = [('jetplane', '20000001'), ('straightwing', '20000000'), ('beanchair', '20000018'),
         ('tulipchair', '20000023'), ('clubchair', '20000027'), ('roundtable', '20000038'),
         ('headboardbeds', '20000004'), ('boxcabinet', '20000009'), ('headlessbeds', '20000005'),
         ('kingsizebeds', '20000006'), ('wassilychair', '20000024'),
         ('cantileverchair', '20000020'), ('shorttable', '20000039'), ('jetplane', '20000000'),
         ('transportairplane', '20000002'), ('miscbeds', '20000007'),
         ('doublecouch', '20000028'), ('barcelonachair', '20000016'), ('bedcabinet', '20000008'),
         ('chaise', '20000026'), ('zigzagchair', '20000026'), ('two-doorcabinet', '20000013'),
         ('sweptwing', '20000001'), ('deskcabinet', '20000010'), ('sidetable', '20000040'),
         ('NO.14chair', '20000021'), ('chaise', '20000022'), ('jet-propelledplane', '20000002'),
         ('rexchair', '20000022'), ('jetplane', '20000002'), ('leathercouch', '20000030'),
         ('rectangulartable', '20000037'), ('ballchair', '20000015'),
         ('garagecabinet', '20000011'), ('workshoptable', '20000041'),
         ('cabinettable', '20000036'), ('tallcabinet', '20000012'),
         ('L-shapedcouch', '20000029'), ('butterflychair', '20000019'), ('Xchair', '20000025')]
    syns = list(wn.all_synsets())
    wn_id_to_synset = {str(s.offset()).zfill(8): s for s in syns}
    for cat, wnid in additional_cats:
        wn_id_to_synset[wnid] = cat
    return wn_id_to_synset

def map_wnid_to_nyu13():
    nyu_13_classes = [(0,'Unknown'),
                      (1,'Bed'),
                      (2,'Books'),
                      (3,'Ceiling'),
                      (4,'Chair'),
                      (5,'Floor'),
                      (6,'Furniture'),
                      (7,'Objects'),
                      (8,'Picture'),
                      (9,'Sofa'),
                      (10,'Table'),
                      (11,'TV'),
                      (12,'Wall'),
                      (13,'Window')
    ]
    mapping = {
        '04593077':4, '03262932':4, '02933112':6, '03207941':7, '03063968':10, '04398044':7, '04515003':7,
        '00017222':7, '02964075':10, '03246933':10, '03904060':10, '03018349':6, '03786621':4, '04225987':7,
        '04284002':7, '03211117':11, '02920259':1, '03782190':11, '03761084':7, '03710193':7, '03367059':7,
        '02747177':7, '03063599':7, '04599124':7, '20000036':10, '03085219':7, '04255586':7, '03165096':1,
        '03938244':1, '14845743':7, '03609235':7, '03238586':10, '03797390':7, '04152829':11, '04553920':7,
        '04608329':10, '20000016':4, '02883344':7, '04590933':4, '04466871':7, '03168217':4, '03490884':7,
        '04569063':7, '03071021':7, '03221720':12, '03309808':7, '04380533':7, '02839910':7, '03179701':10,
        '02823510':7, '03376595':4, '03891251':4, '03438257':7, '02686379':7, '03488438':7, '04118021':5,
        '03513137':7, '04315948':7, '03092883':10, '15101854':6, '03982430':10, '02920083':1, '02990373':3,
        '03346455':12, '03452594':7, '03612814':7, '06415419':7, '03025755':7, '02777927':12, '04546855':12,
        '20000040':10, '20000041':10, '04533802':7, '04459362':7, '04177755':9, '03206908':7, '20000021':4,
        '03624134':7, '04186051':7, '04152593':11, '03643737':7, '02676566':7, '02789487':6, '03237340':6,
        '04502670':7, '04208936':7, '20000024':4, '04401088':7, '04372370':12, '20000025':4, '03956922':7,
        '04379243':10, '04447028':7, '03147509':7, '03640988':7, '03916031':7, '03906997':7, '04190052':6,
        '02828884':4, '03962852':1, '03665366':7, '02881193':7, '03920867':4, '03773035':12, '03046257':12,
        '04516116':7, '00266645':7, '03665924':7, '03261776':7, '03991062':7, '03908831':7, '03759954':7,
        '04164868':7, '04004475':7, '03642806':7, '04589593':13, '04522168':7, '04446276':7, '08647616':4,
        '02808440':7, '08266235':10, '03467517':7, '04256520':9, '04337974':7, '03990474':7, '03116530':6,
        '03649674':4, '04349401':7, '01091234':7, '15075141':7, '20000028':9, '02960903':7, '04254009':7,
        '20000018':4, '20000020':4, '03676759':11, '20000022':4, '20000023':4, '02946921':7, '03957315':7,
        '20000026':4, '20000027':4, '04381587':10, '04101232':7, '03691459':7, '03273913':7, '02843684':7,
        '04183516':7, '04587648':13, '02815950':3, '03653583':6, '03525454':7, '03405725':6, '03636248':7,
        '03211616':11, '04177820':4, '04099969':4, '03928116':7, '04586225':7, '02738535':4, '20000039':10,
        '20000038':10, '04476259':7, '04009801':11, '03909406':12, '03002711':7, '03085602':11, '03233905':6,
        '20000037':10, '02801938':7, '03899768':7, '04343346':7, '03603722':7, '03593526':7, '02954340':7,
        '02694662':7, '04209613':7, '02951358':7, '03115762':9, '04038727':6, '03005285':7, '04559451':7,
        '03775636':7, '03620967':10, '02773838':7, '20000008':6, '04526964':7, '06508816':7, '20000009':6,
        '03379051':7, '04062428':7, '04074963':7, '04047401':7, '03881893':13, '03959485':7, '03391301':7,
        '03151077':12, '04590263':13, '20000006':1, '03148324':6, '20000004':1, '04453156':7, '02840245':2,
        '04591713':7, '03050864':7, '03727837':5, '06277280':11, '03365592':5, '03876519':8, '03179910':7,
        '06709442':7, '03482252':7, '04223580':7, '02880940':7, '04554684':7, '20000030':9, '03085013':7,
        '03169390':7, '04192858':7, '20000029':9, '04331277':4, '03452741':7, '03485997':7, '20000007':1,
        '02942699':7, '03231368':10, '03337140':7, '03001627':4, '20000011':6, '20000010':6, '20000013':6,
        '04603729':10, '20000015':4, '04548280':12, '06410904':2, '04398951':10, '03693474':9, '04330267':7,
        '03015149':9, '04460038':7, '03128519':7, '04306847':7, '03677231':7, '02871439':6, '04550184':6,
        '14974264':7, '04344873':9, '03636649':7, '20000012':6, '02876657':7, '03325088':7, '04253437':7,
        '02992529':7, '03222722':12, '04373704':4, '02851099':13, '04061681':10, '04529681':7,
    }
    english_mapping = {}
    for key, value in mapping.iteritems():
        english_mapping[key] = nyu_13_classes[value]
    return english_mapping

def get_all_semantic_categories():
    load_wnids_for_physics_objects = '/home/bjm113/Research/chrono/canon_textfiles/ShapeNetsSUNRGBDMapping.txt'
    wnids = set()
    with open(load_wnids_for_physics_objects,'r') as f:
        lines = f.readlines()
        for line in lines:
            wnids.add(line.split()[0])
    load_wnids_for_layouts = '/home/bjm113/ScenenetLayouts/scenenet_mat_to_wnid.txt'
    with open(load_wnids_for_layouts,'r') as f:
        lines = f.readlines()
        for line in lines:
            wnids.add(line.split()[1])
    return wnids


semantic_categories = get_all_semantic_categories()
print('Semantic Categories')
print len(semantic_categories)
semantic_wnid_lookup = get_id_to_wnsynset_dict()

nyu13_mapping = map_wnid_to_nyu13()
for category in semantic_categories:
    if type(semantic_wnid_lookup[category]) == type('string'):
        print '{0:<30} -> {1:<15}'.format(semantic_wnid_lookup[category],nyu13_mapping[category][1])
    else:
        print '{0:<30} -> {1:<15}'.format(semantic_wnid_lookup[category].name(),nyu13_mapping[category][1])
```     
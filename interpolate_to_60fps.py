import os
import sys
import numpy as np
from absl import app
from absl import flags
import pickle
from utils.utils import *

FLAGS = flags.FLAGS
flags.DEFINE_string('input_motion',"./temp", "path to 3d keypoints + extension")
flags.DEFINE_string('output_dir',"./temp", "path to 3d keypoints + extension")

def interpolate(np_frames):

   frames = np_frames.tolist()

   new_frames = []
   for i in range(len(frames) - 1):
      second_points = np.zeros((17, 3))
      third_points = np.zeros((17, 3))
      left_points = frames[i]
      right_points = frames[i + 1]

      for j in range(len(second_points)):

         delta_x = (right_points[j][0] - left_points[j][0])/3
         delta_y = (right_points[j][1] - left_points[j][1])/3
         delta_z = (right_points[j][2] - left_points[j][2])/3

         second_points[j][0] = left_points[j][0] + delta_x
         second_points[j][1] = left_points[j][1] + delta_y
         second_points[j][2] =left_points[j][2] + delta_z

      for j in range(len(third_points)):
         third_points[j][0] = (left_points[j][0] + right_points[j][0])/2
         third_points[j][1] = (left_points[j][1] + right_points[j][1])/2
         third_points[j][2] = (left_points[j][2] + right_points[j][2])/2

         delta_x = (right_points[j][0] - left_points[j][0])/3
         delta_y = (right_points[j][1] - left_points[j][1])/3
         delta_z = (right_points[j][2] - left_points[j][2])/3

         third_points[j][0] = left_points[j][0] + (2*delta_x)
         third_points[j][1] = left_points[j][1] + (2*delta_y)
         third_points[j][2] =left_points[j][2] + (2*delta_z)

      new_frames.append(left_points)
      new_frames.append(second_points)
      new_frames.append(third_points)

   new_frames.append(frames[-1])
   new_frames.append(frames[-1])
   new_frames.append(frames[-1])

   return new_frames


def main(_):

   motion = np.load(FLAGS.input_motion, allow_pickle=True)

   # interpolated = interpolate(motion)

   # save_obj(interpolated, FLAGS.output_dir, name )

if __name__ == '__main__':

   if not os.path.exists(args.output_dir):
        os.mkdir(FLAGS.output_dir)

   app.run(main)

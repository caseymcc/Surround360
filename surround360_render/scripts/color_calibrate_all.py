import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import threading
import time
from timeit import default_timer as timer

script_dir = os.path.dirname(os.path.realpath(__file__))

# os.path.dirname(DIR) is the parent directory of DIR
surround360_render_dir = os.path.dirname(script_dir)

TITLE = "Surround 360 - Color Calibration"

COLOR_CALIBRATION_COMMAND_TEMPLATE = """
{SURROUND360_RENDER_DIR}/bin/TestColorCalibration
--image_path {IMAGE_PATH}
--isp_passthrough_path {ISP_JSON}
--output_data_dir {OUTPUT_DIR}
--log_dir {LOG_DIR}
--clamp_min {CLAMP_MIN}
--clamp_max {CLAMP_MAX}
--logbuflevel -1
--stderrthreshold 0
{FLAGS_EXTRA}
"""

def list_tiff(src_dir): return [os.path.join(src_dir, fn) for fn in next(os.walk(src_dir))[2] if fn.endswith('.tiff')]

def parse_args():
  parse_type = argparse.ArgumentParser
  parser = parse_type(description=TITLE, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--data_dir',             help='directory containing raw calibration images', required=True)
  parser.add_argument('--output_dir',           help='output directory', required=False)
  parser.add_argument('--black_level_darkest',  help='if true, assume black level is darkest point', action='store_true')
  parser.add_argument('--black_level_adjust',   help='if true, sets each channel black level to median of all cameras', action='store_true')
  parser.add_argument('--black_level',          help='manual black level', required=False, default='NONE')
  return vars(parser.parse_args())

def start_subprocess(name, cmd):
  global current_process
  current_process = subprocess.Popen(cmd, shell=True)
  current_process.name = name
  current_process.communicate()

def print_and_save(file_out, str):
  print str
  file_out.write(str)
  sys.stdout.flush()

def save_step_runtime(file_out, step, runtime_sec):
  text_runtime = "\n" + step + " runtime: " + str(datetime.timedelta(seconds=runtime_sec)) + "\n"
  file_out.write(text_runtime)
  print text_runtime
  sys.stdout.flush()

def run_step(step, cmd, file_runtimes):
  file_runtimes.write("\n" + cmd + "\n")
  print cmd + "\n"
  sys.stdout.flush()
  start_time = timer()
  start_subprocess(step, cmd)
  save_step_runtime(file_runtimes, step, timer() - start_time)

def run_threads(thread_list):
  for thread in thread_list:
    thread.start()
  for thread in thread_list:
    thread.join()

def median(list):
  q, r = divmod(len(list), 2)
  return sorted(list)[q] if r else sum(sorted(list)[q - 1:q + 1]) / 2.0

if __name__ == "__main__":
  args = parse_args()
  data_dir = args["data_dir"]
  black_level_darkest = args["black_level_darkest"]
  black_level_adjust = args["black_level_adjust"]
  black_level = args["black_level"]

  print "\n--------" + time.strftime(" %a %b %d %Y %H:%M:%S %Z ") + "-------\n"

  os.chdir(surround360_render_dir)

  if args["output_dir"] is not None:
    out_dir = args["output_dir"]
  else:
    out_dir = data_dir + "/output"
  os.system("mkdir -p " + out_dir)

  isp_dir = data_dir + "/isp"
  os.system("mkdir -p " + isp_dir)

  file_runtimes = open(out_dir + "/runtimes.txt", 'w', 0)
  start_time = timer()

  isp_passthrough_json = surround360_render_dir + "/res/config/isp/passthrough.json"
  raw_charts = list_tiff(data_dir + "/charts")

  ### First find all the X-intercepts ###

  step = "first_pass"
  print step + "\n"

  flags_extra = ""
  if black_level_darkest:
    flags_extra += " --black_level_darkest"
  elif black_level != 'NONE':
    flags_extra += " --black_level \"" + black_level + "\""

  out_dirs = {}
  camera_names = {}
  thread_list = []
  for i in range(len(raw_charts)):
    camera_names[i] = os.path.basename(raw_charts[i]).split('_')[0]
    out_dirs[i] = out_dir + "/" + camera_names[i]
    color_calibrate_params = {
      "SURROUND360_RENDER_DIR": surround360_render_dir,
      "IMAGE_PATH": raw_charts[i],
      "ISP_JSON": isp_passthrough_json,
      "OUTPUT_DIR": out_dirs[i] + "_" + step,
      "LOG_DIR": out_dirs[i]+ "_" + step,
      "CLAMP_MIN": 0.0,
      "CLAMP_MAX": 1.0,
      "FLAGS_EXTRA": flags_extra + " --just_intercepts",
    }
    color_calibrate_command = COLOR_CALIBRATION_COMMAND_TEMPLATE.replace("\n", " ").format(**color_calibrate_params)
    t = threading.Thread(target=run_step, args=(camera_names[i] + " first pass", color_calibrate_command, file_runtimes,))
    thread_list.append(t)

  run_threads(thread_list)

  ### Adapt all cameras to worst dynamic range ###

  print "Finding worst-case X-intercepts...\n"

  intercept_x_min = float("inf")
  intercept_x_max = -float("inf")
  print_and_save(file_runtimes, "\n")
  for i in range(len(out_dirs)):
    intercepts = json.loads(open(out_dirs[i] + "_" + step + "/intercept_x.txt").read())
    print_and_save(file_runtimes, camera_names[i] + ": " + str(intercepts) + "\n")

    intercept_x_max = max(intercept_x_max, max(intercepts[0]));
    intercept_x_min = min(intercept_x_min, min(intercepts[1]));

  text_intercepts = ("Intercept Xmin max: " + str(intercept_x_max) + ", " +
                     "Intercept Xmax min: " + str(intercept_x_min) + "\n")
  print_and_save(file_runtimes, text_intercepts)

  ### Adapt all cameras to same per-channel black level (median) ###

  if black_level_adjust:
    print "Adjusting black levels...\n"

    black_levels = {}
    num_channels = 3
    for j in range(num_channels):
      black_levels[j] = [];

    for i in range(len(out_dirs)):
      black_level = json.loads(open(out_dirs[i] + "_" + step + "/black_level.txt").read())
      print_and_save(file_runtimes, camera_names[i] + ": " + str(black_level) + "\n")
      for j in range(num_channels):
        black_levels[j].append(black_level[j]);

    black_level_median = []
    for j in range(num_channels):
      black_level_median.append(median(black_levels[j]));

    print "Black level median: " + str(black_level_median)
    flags_extra += " --black_level \"" + " ".join(map(str, black_level_median)) + "\""

  step = "second_pass"
  print "\n" + step + "\n"
  flags_extra += " --save_debug_images"

  thread_list = []
  for i in range(len(raw_charts)):
    color_calibrate_params = {
      "SURROUND360_RENDER_DIR": surround360_render_dir,
      "IMAGE_PATH": raw_charts[i],
      "ISP_JSON": isp_passthrough_json,
      "OUTPUT_DIR": out_dirs[i]+ "_" + step,
      "LOG_DIR": out_dirs[i]+ "_" + step,
      "CLAMP_MIN": intercept_x_max,
      "CLAMP_MAX": intercept_x_min,
      "FLAGS_EXTRA": flags_extra,
    }
    color_calibrate_command = COLOR_CALIBRATION_COMMAND_TEMPLATE.replace("\n", " ").format(**color_calibrate_params)
    t = threading.Thread(target=run_step, args=(camera_names[i] + " second pass", color_calibrate_command, file_runtimes,))
    thread_list.append(t)

  run_threads(thread_list)

  for i in range(len(raw_charts)):
    camera_number = re.findall("cam(\d+)", camera_names[i])[0]
    isp_src = out_dirs[i]+ "_" + step + "/isp_out.json"
    isp_dst = isp_dir + "/isp" + camera_number + ".json"

    print "Copying " + isp_src + " to " + isp_dst + "..."
    os.system("cp " + isp_src + " " + isp_dst)

  save_step_runtime(file_runtimes, "TOTAL", timer() - start_time)
  file_runtimes.close()

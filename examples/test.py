import os
from re import I
import quaternion
import habitat_sim.sim
import numpy as np
from scipy.io import wavfile
import random
import matplotlib.pyplot as plt
from habitat.utils.visualizations import maps
import imageio
from scipy.spatial.transform import Rotation as R
import math
from pyroomacoustics.experimental.rt60 import measure_rt60
from scipy.signal import fftconvolve
from pydub import AudioSegment

import soundfile as sf
from scipy import signal
import torch
import torch.nn.functional as F
import sys

from pathlib import Path
sys.path.append("/content/Spatial_AST")
import spatial_ast

def normalize_audio(audio_data, target_dBFS=-14.0):
    rms = np.sqrt(np.mean(audio_data**2)) # Calculate the RMS of the audio
   
    if rms == 0:  # Avoid division by zero in case of a completely silent audio
        return audio_data
    
    current_dBFS = 20 * np.log10(rms) # Convert RMS to dBFS
    gain_dB = target_dBFS - current_dBFS # Calculate the required gain in dB
    gain_linear = 10 ** (gain_dB / 20) # Convert gain from dB to linear scale
    normalized_audio = audio_data * gain_linear # Apply the gain to the audio data
    return normalized_audio

device = torch.device('cuda')

os.chdir('/content/sound-spaces')
os.makedirs('/content/output', exist_ok=True)
dataset = 'mp3d'

backend_cfg = habitat_sim.SimulatorConfiguration()

if dataset == 'mp3d':
    backend_cfg.scene_id = "data/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb"
    # IMPORTANT: missing this file will lead to load the semantic scene incorrectly
    backend_cfg.scene_dataset_config_file = "data/scene_datasets/mp3d/mp3d.scene_dataset_config.json"
else:
    backend_cfg.scene_id = "data/scene_datasets/gibson/Oyens.glb"
    # IMPORTANT: missing this file will lead to load the semantic scene incorrectly
    backend_cfg.scene_dataset_config_file = "data/scene_datasets/gibson/gibson_semantic.scene_dataset_config.json"
backend_cfg.load_semantic_mesh = True
backend_cfg.enable_physics = False
agent_config = habitat_sim.AgentConfiguration()
cfg = habitat_sim.Configuration(backend_cfg, [agent_config])
sim = habitat_sim.Simulator(cfg)

seed = random.randint(0, 100)
print('seed', seed)
#sim.seed(seed)

# set navmesh path for searching for navigable points
if dataset == 'mp3d':
    sim.pathfinder.load_nav_mesh(os.path.join(f"data/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.navmesh"))
else:
    sim.pathfinder.load_nav_mesh(os.path.join(f"data/scene_datasets/gibson/Oyens.navmesh"))

audio_sensor_spec = habitat_sim.AudioSensorSpec()
audio_sensor_spec.uuid = "audio_sensor"
audio_sensor_spec.enableMaterials = True
audio_sensor_spec.channelLayout.type = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Binaural
audio_sensor_spec.channelLayout.channelCount = 1
# audio sensor location set with respect to the agent
audio_sensor_spec.position = [0.0, 1.5, 0.0]  # audio sensor has a height of 1.5m
audio_sensor_spec.acousticsConfig.sampleRate = 48000
# whether indrect (reverberation) is present in the rendered IR
audio_sensor_spec.acousticsConfig.indirect = True
sim.add_sensor(audio_sensor_spec)

audio_sensor = sim.get_agent(0)._sensors["audio_sensor"]
audio_sensor.setAudioMaterialsJSON("data/mp3d_material_config.json")

# sampled navigable point is on the floor
#source_pos = sim.pathfinder.get_random_navigable_point()
#print('Sample source location: ', source_pos)
source_pos = np.array([-7.9259,1.52,-2.8804])
#source_pos = np.array([-8.2144,1.1012,-4.3399])
#source_pos = np.array([-9.3071,1.9991,-2.8131])
#source_pos = np.array([-3.4029,0.9901,-0.1906])

#audio_sensor.setAudioSourceTransform(source_pos + np.array([0, 1.5, 0])) # add 1.5m to the height calculation 
audio_sensor.setAudioSourceTransform(source_pos)
agent = sim.get_agent(0)
new_state = sim.get_agent(0).get_state()
#new_state.position = sim.pathfinder.get_random_navigable_point_near(source_pos, 2)
new_state.position = np.array([-7.2797,0.0724,-2.0944])
#new_state.position = np.array([-7.5873,0.0724,-1.7346])
#new_state.position = np.array([-4.0593,0.0724,-0.311])

print('agent position', new_state.position)
new_state.sensor_states = {}
agent.set_state(new_state, True)
#ir = np.array(sim.get_sensor_observations()["audio_sensor"])
ir = np.load('/content/drive/MyDrive/mp3d_reverb/binaural/17DRP5sb8fy/0.npy')
print('ir shape', ir.shape)

# check if the direct sound is present (source is visibile from the listener)
print('source is visble', audio_sensor.sourceIsVisible())

# check the efficiency of rendering, outdoor would have a very low value, e.g. < 0.05, 
# while a closed indoor room would have >0.95, and a room with some holes might be in the 0.1-0.8 range.
# if the ray efficiency is low for an indoor environment, it indicates a lot of ray leak from holes
# you should repair the mesh in this case for more accurate acoustic rendering
print('ray efficiency', audio_sensor.getRayEfficiency())

# plot the waveform of IR and show the audio
sr = 48000
samples_clip = 10000
plt.title('Impulse Response')
for i in range(2):
    plt.plot(np.linspace(0, samples_clip / sr, samples_clip), ir[i, :samples_clip])
    plt.savefig(os.path.join('/content', 'output', 'ir{}'.format(i)))

plt.clf()



# convert input audio from stereo to mono
sound = AudioSegment.from_wav('/content/drive/MyDrive/speech_navigation/follow_me.wav')
sound = sound.set_channels(1)
sound.export("/content/follow_me_mono.wav", format="wav")

sr, vocal = wavfile.read('/content/follow_me_mono.wav')
print('sr', sr, 'vocal shape', vocal.shape)

plt.title('Original Speech')
plt.plot(np.linspace(0, len(vocal) / sr, len(vocal)), vocal)
plt.savefig(os.path.join('/content', 'output', 'original_speech'))
plt.clf()


# convolve the vocal with IR
convolved_vocal = np.array([fftconvolve(vocal, ir_channel) for ir_channel in ir])
print('convolved_vocal.shape', convolved_vocal.shape)
plt.title('Speech convolved with IR')
for i in range(2):
    plt.plot(np.linspace(0, len(convolved_vocal[i]) / sr, len(convolved_vocal[i])), convolved_vocal[i])
    plt.savefig(os.path.join('/content', 'output', 'speech_convolved_with_ir{}'.format(i)))
plt.clf()

#convolved_vocal = convolved_vocal / np.expand_dims(np.max(convolved_vocal, axis=1), axis=1)
print('max(convolved_vocal) before', np.max(convolved_vocal))
print('min(convolved_vocal) before', np.min(convolved_vocal))

iinfo = np.iinfo(np.int16)
convolved_vocal = (convolved_vocal / np.max(np.abs(convolved_vocal)) * iinfo.max).astype(np.int16)
print('max(convolved_vocal) after', np.max(convolved_vocal))
print('min(convolved_vocal) after', np.min(convolved_vocal))
#convolved_vocal = (convolved_vocal / np.max(convolved_vocal) * np.iinfo(np.int16).max).astype(np.int16)
os.makedirs('/content/output/audio/reverberent_speech', exist_ok=True)
wavfile.write(os.path.join('/content', 'output', 'audio', 'reverberent_speech', '0.wav'), sr, convolved_vocal.T)



rt60 = measure_rt60(ir[0], sr, decay_db=30, plot=True)
print(f'RT60 of the rendered IR is {rt60:.4f} seconds')

action_names = list(cfg.agents[0].action_space.keys())
print('action names', action_names)

# convert 3d points to 2d topdown pixel coordinates
def convert_points_to_topdown_pixel_coordinates(pathfinder, points, meters_per_pixel):
    points_topdown = []
    bounds = pathfinder.get_bounds()
    for point in points:
        # convert 3D x,z to topdown x,y
        px = (point[0] - bounds[0][0]) / meters_per_pixel
        py = (point[2] - bounds[0][2]) / meters_per_pixel
        points_topdown.append(np.array([px, py]))
    return points_topdown


# display a topdown map with matplotlib
def display_map(topdown_map, out_file, source_pos=None, agent_pos=None, agent_angle=None):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    #ax.axis("off")
    plt.imshow(topdown_map)
    # plot points on map
    if source_pos is not None:
        plt.plot(source_pos[0], source_pos[1], marker="o", markersize=10, alpha=0.8)
    if agent_pos is not None:
        plt.plot(agent_pos[0], agent_pos[1], marker="o", markersize=10, alpha=0.8)
    if agent_angle is not None:
        arrow_len = 50
        arrow_dest = (agent_pos[0] + arrow_len * math.sin(agent_angle), agent_pos[1] - arrow_len * math.cos(agent_angle))
        ax.annotate('', xytext=agent_pos, xy=arrow_dest, arrowprops=dict(arrowstyle="->"))


    #plt.show(block=False)
    plt.savefig(os.path.join('/content', 'output', out_file))

def quaternion_to_axis_angle(quat):
    q_array = np.array([quat.x, quat.y, quat.z, quat.w])
    rotation = R.from_quat(q_array)
    
    # Get the axis-angle representation
    axis_angle = rotation.as_rotvec()
    
    # Calculate the angle from the magnitude of the rotation vector
    angle = np.linalg.norm(axis_angle)
    
    # Normalize the rotation vector to get the axis
    if angle != 0:
        axis = axis_angle / angle
    else:
        axis = np.array([1.0, 0.0, 0.0])  # Handle zero rotation case

    return axis, angle

meters_per_pixel = 0.01  # @param {type:"slider", min:0.01, max:1.0, step:0.01}
# @markdown Customize the map slice height (global y coordinate):
custom_height = False  # @param {type:"boolean"}
height = 1

bounds = sim.pathfinder.get_bounds()
print("The NavMesh bounds are: " + str(bounds))
if not custom_height:
    # get bounding box minimum elevation for automatic height
    height = sim.pathfinder.get_bounds()[0][1]

if not sim.pathfinder.is_loaded:
    print("Pathfinder not initialized, aborting.")
else:
    # @markdown You can get the topdown map directly from the Habitat-sim API with *PathFinder.get_topdown_view*.
    # This map is a 2D boolean array
    sim_topdown_map = sim.pathfinder.get_topdown_view(meters_per_pixel, height)
    
    # @markdown Alternatively, you can process the map using the Habitat-Lab [maps module](https://github.com/facebookresearch/habitat-lab/blob/main/habitat/utils/visualizations/maps.py)
    hablab_topdown_map = maps.get_topdown_map(
        sim.pathfinder, height, meters_per_pixel=meters_per_pixel
    )
    recolor_map = np.array(
        [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
    )
    hablab_topdown_map = recolor_map[hablab_topdown_map]

    agent_rot_quat = agent.get_state().rotation
    agent_pos = agent.get_state().position
    pos_pixels = convert_points_to_topdown_pixel_coordinates(sim.pathfinder, [source_pos, agent_pos], meters_per_pixel)
    print('source position', source_pos)
    print('agent position', agent_pos)
    print('agent rotation quaternion', agent_rot_quat)
    rot_axis, rot_angle = quaternion_to_axis_angle(agent_rot_quat)
    print('rot_axis', rot_axis)
    print('rot_angle', rot_angle)
    print('source and agent pixel positions', pos_pixels)

    print("Displaying the raw map from get_topdown_view:")
    display_map(sim_topdown_map, 'habitat_sim_get_topdown_view', source_pos=pos_pixels[0], agent_pos=pos_pixels[1], agent_angle=rot_angle)
    print("Displaying the map from the Habitat-Lab maps module:")
    display_map(hablab_topdown_map, 'habitat_lab_get_topdown_map', source_pos=pos_pixels[0], agent_pos=pos_pixels[1], agent_angle=rot_angle)
    

print('reverb padding start')
print('ir.shape before', ir.shape)
reverb_padding = 32000 * 2 - ir.shape[1]
ir = torch.from_numpy(ir).float()
if reverb_padding > 0:
    ir = F.pad(ir, (0, reverb_padding), 'constant', 0)
elif reverb_padding < 0:
    ir = ir[:, :32000 * 2]
print('ir.shape after', ir.shape)

normalize = True
waveform, sr = sf.read('/content/drive/MyDrive/speech_navigation/follow_me.wav')
waveform = waveform[:, 0] if len(waveform.shape) > 1 else waveform
waveform = signal.resample_poly(waveform, 32000, sr) if sr != 32000 else waveform   
waveform = normalize_audio(waveform, -14.0) if normalize else waveform

waveform = torch.from_numpy(waveform).reshape(1, -1).float()
# We pad all audio samples into 10 seconds long
padding = 32000 * 10 - waveform.shape[1]
if padding > 0:
    waveform = F.pad(waveform, (0, padding), 'constant', 0)
elif padding < 0:
    waveform = waveform[:, :32000 * 10]

model = spatial_ast.__dict__['build_AST'](
    num_classes=355,
    drop_path_rate=0.1,
    num_cls_tokens=3,
)

checkpoint = torch.load('/content/drive/MyDrive/finetuned.pth', map_location='cpu')
print('Load pre-trained checkpoint')
checkpoint_model = checkpoint['model']
msg = model.load_state_dict(checkpoint_model, strict=False)
print(msg)

model.to(device)
waveform.to(device)
ir.to(device)

output = model(torch.unsqueeze(waveform, 0).cuda(), torch.unsqueeze(ir, 0).cuda())
distance_logits = output[1].detach().cpu().numpy()
azimuth_logits = output[2].detach().cpu().numpy()
elevation_logits = output[3].detach().cpu().numpy()

print('len(distance logits)', len(distance_logits[0]))
print('len(azimuth logits)', len(azimuth_logits[0]))
print('len(elevation logits)', len(elevation_logits[0]))

plt.clf()
plt.bar(np.arange(len(distance_logits[0])), distance_logits[0])
plt.savefig(os.path.join('/content', 'output', 'distance_logits'))

plt.clf()
plt.bar(np.arange(len(azimuth_logits[0])), azimuth_logits[0])
plt.savefig(os.path.join('/content', 'output', 'azimuth_logits'))

plt.clf()
plt.bar(np.arange(len(elevation_logits[0])), elevation_logits[0])
plt.savefig(os.path.join('/content', 'output', 'elevation_logits'))

dx = source_pos[0] - new_state.position[0] # LEFT-RIGHT
dy = source_pos[1] - (new_state.position[1] + 1.5) # UP-DOWN
dz = source_pos[2] - new_state.position[2] # FRONT-BACK

print('distance prediction', np.argmax(distance_logits, axis=1))
print('azimuth prediction', np.argmax(azimuth_logits, axis=1))
azimuth_gt = math.degrees(math.atan2(-dz, dx))
azimuth_gt = (round(azimuth_gt) + 360) % 360
print('azimuth GT', azimuth_gt)
print('elevation prediction', np.argmax(elevation_logits, axis=1))

print('done spatial ast')

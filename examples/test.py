import os
from re import I
import json
import quaternion
import habitat_sim
import habitat_sim.sim
import habitat
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
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

ir_type = 'ir_precomputed' # 'ir_precomputed', 'ir_computed'

def normalize_audio(audio_data, target_dBFS=-14.0):
    rms = np.sqrt(np.mean(audio_data ** 2))  # Calculate the RMS of the audio

    if rms == 0:  # Avoid division by zero in case of a completely silent audio
        return audio_data

    current_dBFS = 20 * np.log10(rms)  # Convert RMS to dBFS
    gain_dB = target_dBFS - current_dBFS  # Calculate the required gain in dB
    gain_linear = 10 ** (gain_dB / 20)  # Convert gain from dB to linear scale
    normalized_audio = audio_data * gain_linear  # Apply the gain to the audio data
    return normalized_audio


def run_spatial_ast(ir, scene_id, setup_id):
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
    waveform, sr = sf.read('/content/drive/MyDrive/speech_navigation/come_here_10s.wav')    
    #waveform, sr = sf.read('/content/drive/MyDrive/data_and_checkpoints/eval/audio/YSObG51xihGc.wav')
    #waveform, sr = sf.read('/content/sound-spaces/res/singing.wav')
    #waveform, sr = sf.read('/content/drive/MyDrive/speech_navigation/follow_me.wav')
    print('DEBUG sr', sr)
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

    agent_pos = agent.get_state().position
    print('agent_pos', agent_pos)
    print('source_pos', source_pos)

    dx = source_pos[0] - agent_pos[0]  # LEFT-RIGHT
    dy = source_pos[1] - (agent_pos[1] + 1.5)  # UP-DOWN
    dz = source_pos[2] - agent_pos[2]  # FRONT-BACK
    print('dy', dy)
    distance_gt = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    print('distance GT', distance_gt)
    dist_pred = np.argmax(distance_logits, axis=1)[0] * 0.5
    azimuth_pred = np.argmax(azimuth_logits, axis=1)[0]
    print('distance prediction', dist_pred)
    print('azimuth prediction', azimuth_pred)
    azimuth_gt = math.degrees(math.atan2(-dz, dx))
    azimuth_gt = (round(azimuth_gt) + 360) % 360
    print('azimuth GT', azimuth_gt)
    elevation_pred = np.argmax(elevation_logits, axis=1)[0]
    elevation_gt = math.degrees(math.atan(dy / math.sqrt(dx**2 + dz**2)))
    elevation_gt = (round(elevation_gt) + 90) % 180
    #elevation_pred = 25
    print('elevation prediction', elevation_pred)
    print('elevation GT', elevation_gt)

    plt.clf()
    plt.title('Sound Distance Estimation')
    plt.bar(np.arange(0, len(distance_logits[0]) / 2, 0.5), distance_logits[0], width=0.3)
    plt.vlines(distance_gt, abs(min(distance_logits[0])) * -1.25, max(distance_logits[0]) * 1.25, colors='green')
    #plt.plot(distance_gt, np.max(distance_logits[0]) * 1.5, 'go', markersize=15)
    plt.legend(['Ground Truth Distance', 'Distance Prediction Logits'])
    plt.xlabel('Distance (m)')
    plt.savefig(os.path.join('/content/drive/MyDrive/speech_navigation_output', scene_id, setup_id, ir_type, 'distance_logits'))

    plt.clf()
    plt.title('Sound Azimuth Estimation')
    plt.bar(np.arange(len(azimuth_logits[0])), azimuth_logits[0], width=0.3)
    plt.vlines(azimuth_gt, abs(min(azimuth_logits[0])) * -1.25, max(azimuth_logits[0]) * 1.25, colors='green')
    #plt.plot(azimuth_gt, np.max(azimuth_logits[0]) * 1.5, 'go', markersize=10)
    plt.legend(['Ground Truth Azimuth', 'Azimuth Prediction Logits'])
    plt.xlabel('Angle (Degrees)')
    plt.savefig(os.path.join('/content/drive/MyDrive/speech_navigation_output', scene_id, setup_id, ir_type, 'azimuth_logits'))

    plt.clf()
    plt.title('Sound Elevation Estimation')
    plt.bar(np.arange(len(elevation_logits[0])), elevation_logits[0], width=0.3)
    plt.vlines(elevation_gt, abs(min(elevation_logits[0])) * -1.25, max(elevation_logits[0]) * 1.25, colors='green')
    plt.legend(['Ground Truth Elevation', 'Elevation Prediction Logits'])
    
    plt.xlabel('Angle (Degrees)')
    plt.savefig(os.path.join('/content/drive/MyDrive/speech_navigation_output', scene_id, setup_id, ir_type, 'elevation_logits'))

    return dist_pred, elevation_pred, azimuth_pred, distance_gt, azimuth_gt, elevation_gt


def get_shortest_path(sim, agent, goal_pos):
    path = habitat_sim.ShortestPath()
    path.requested_start = agent_pos
    path.requested_end = goal_pos

    if sim.pathfinder.find_path(path):
        print('path points', path.points)

    follower = habitat_sim.nav.GreedyGeodesicFollower(
        sim.pathfinder,
        agent,
        goal_radius=0.25
        # stop_key='stop',
        # forward_key='move_forward',
        # left_key='left_key',
        # right_key='right_key'
    )

    # print(follower.find_path(goal_pos))
    path_points = []
    cnt = 0
    while cnt < 50:
        path_points.append(agent.get_state().position)
        try:
            next_action = follower.next_action_along(goal_pos)
            if next_action is None:
                break
            print('Next action', next_action)
            sim.step(next_action)
            cnt += 1
        except habitat_sim.errors.GreedyFollowerError:
            print('GreedyFollowerError')
            break
    return path_points


def dist_angle_to_cartesian(agent_pos, dist_pred, elevation_pred, azimuth_pred):
    elevation_pred -= 90
    pred_x = agent_pos[0] + dist_pred * math.cos(math.radians(elevation_pred)) * math.cos(math.radians(azimuth_pred))
    pred_y = agent_pos[1] + dist_pred * math.sin(math.radians(elevation_pred))
    pred_z = agent_pos[2] + dist_pred * math.cos(math.radians(elevation_pred)) * -math.sin(math.radians(azimuth_pred))
    return [pred_x, pred_y, pred_z]


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
def display_map(topdown_map, out_file, scene_id, setup_id, source_pos=None, agent_pos=None, goal_pos=None,
                agent_angle=None, path_points=None):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)
    # plot points on map
    if source_pos is not None:
        plt.plot(source_pos[0], source_pos[1], "go", markersize=10, alpha=0.8)
    if goal_pos is not None:
        plt.plot(goal_pos[0], goal_pos[1], "bo", markersize=10, alpha=0.8)
    if agent_pos is not None:
        plt.plot(agent_pos[0], agent_pos[1], "ro", markersize=10, alpha=0.8)
    if path_points is not None:
        plt.plot([point[0] for point in path_points], [point[1] for point in path_points], "ro", markersize=5,
                 alpha=0.8)

    plt.legend(['GT Speech Source Position', 'Estimated Sound Source Position', 'Agent End Position', 'Agent Steps'])

    if agent_angle is not None:
        arrow_len = 50
        arrow_dest = (
        agent_pos[0] - arrow_len * math.sin(agent_angle), agent_pos[1] - arrow_len * math.cos(agent_angle))
        ax.annotate('', xytext=agent_pos, xy=arrow_dest, arrowprops=dict(arrowstyle="->"))

    # plt.show(block=False)
    # plt.set_xticks([1,2,3,4])
    # plt.tight_layout()
    plt.savefig(os.path.join('/content/drive/MyDrive/speech_navigation_output', scene_id, setup_id, ir_type, out_file),
                bbox_inches='tight')


def quaternion_to_axis_angle(quat):
    q_array = np.array([quat.x, quat.y, quat.z, quat.w])
    rotation = R.from_quat(q_array)

    # Get the axis-angle representation
    axis_angle = rotation.as_rotvec()

    # Calculate the angle from the magnitude of the rotation vector
    angle = np.sign(axis_angle[1]) * np.linalg.norm(axis_angle)

    # Normalize the rotation vector to get the axis
    if angle != 0:
        axis = axis_angle / angle
    else:
        axis = np.array([0.0, 1.0, 0.0])  # Handle zero rotation case

    return axis, angle

def euclidean_dist(x, y):
    return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2 + (x[2] - y[2]) ** 2)


device = torch.device('cuda')

os.chdir('/content/sound-spaces')
os.makedirs('/content/output', exist_ok=True)
dataset = 'mp3d'
scene_id = '17DRP5sb8fy'

backend_cfg = habitat_sim.SimulatorConfiguration()
backend_cfg.scene_id = f"data/scene_datasets/mp3d/{scene_id}/{scene_id}.glb"
# IMPORTANT: missing this file will lead to load the semantic scene incorrectly
backend_cfg.scene_dataset_config_file = "data/scene_datasets/mp3d/mp3d.scene_dataset_config.json"
backend_cfg.load_semantic_mesh = True
backend_cfg.enable_physics = False
agent_config = habitat_sim.AgentConfiguration()
cfg = habitat_sim.Configuration(backend_cfg, [agent_config])
sim = habitat_sim.Simulator(cfg)

audio_sensor_spec = habitat_sim.AudioSensorSpec()
audio_sensor_spec.uuid = "audio_sensor"
audio_sensor_spec.enableMaterials = True
audio_sensor_spec.channelLayout.type = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Binaural
audio_sensor_spec.channelLayout.channelCount = 1
# audio sensor location set with respect to the agent
audio_sensor_spec.position = [0.0, 1.5, 0.0]  # audio sensor has a height of 1.5m
audio_sensor_spec.acousticsConfig.sampleRate = 32000
# whether indrect (reverberation) is present in the rendered IR
audio_sensor_spec.acousticsConfig.indirect = True
sim.add_sensor(audio_sensor_spec)

audio_sensor = sim.get_agent(0)._sensors["audio_sensor"]
audio_sensor.setAudioMaterialsJSON("data/mp3d_material_config.json")

# seed = random.randint(0, 100)
# print('seed', seed)
# sim.seed(seed)

azimuth_in_range = 0
azimuth_error_sum = 0
dist_in_range = 0
dist_error_sum = 0
success_cnt = 0
total_cnt = 0
# set navmesh path for searching for navigable points
sim.pathfinder.load_nav_mesh(os.path.join(f"data/scene_datasets/mp3d/{scene_id}/{scene_id}.navmesh"))

reverb_config = json.load(open('/content/drive/MyDrive/mp3d_reverb/train_reverberation.json'))
for scene_config in reverb_config['data']:
    if not scene_config['fname'].startswith(scene_id):
        continue

    total_cnt += 1

    setup_id = scene_config['fname'].split('/')[-1].split('.')[0]
    print('scene_id', scene_id)
    print('setup_id', setup_id)
    os.makedirs(os.path.join('/content/drive/MyDrive/speech_navigation_output', scene_id, setup_id, ir_type), exist_ok=True)

    # try:
    fname = f'/content/drive/MyDrive/mp3d_reverb/binaural/{scene_config["fname"]}'
    agent_pos = [float(val) for val in scene_config['agent_position'].split(',')]
    sensor_pos = [float(val) for val in scene_config['sensor_position'].split(',')]
    source_pos = [float(val) for val in scene_config['source_position'].split(',')]

    print('source_pos', source_pos)
    print('agent_pos', agent_pos)
    print('sensor_pos', sensor_pos)

    ir_precomputed = np.load(fname)
    print('ir_precomputed.shape', ir_precomputed.shape)

    # source_pos = np.array([-7.9259,1.52,-2.8804])
    # source_pos = np.array([-8.2144,1.1012,-4.3399])
    # source_pos = np.array([-9.3071,1.9991,-2.8131])
    # source_pos = np.array([-3.4029,0.9901,-0.1906])

    audio_sensor.setAudioSourceTransform(source_pos)
    agent = sim.get_agent(0)
    new_state = sim.get_agent(0).get_state()
    new_state.position = np.array(agent_pos)
    # new_state.position = sim.pathfinder.get_random_navigable_point_near(source_pos, 6)
    # new_state.position = np.array([-7.2797,0.0724,-2.0944])
    # new_state.position = np.array([-7.5873,0.0724,-1.7346])
    # new_state.position = np.array([-4.0593,0.0724,-0.311])

    print('agent position', new_state.position)
    new_state.sensor_states = {}
    agent.set_state(new_state, True)

    ir = np.array(sim.get_sensor_observations()["audio_sensor"])
    print('ir shape', ir.shape)

    # check if the direct sound is present (source is visibile from the listener)
    print('source is visble', audio_sensor.sourceIsVisible())

    # check the efficiency of rendering, outdoor would have a very low value, e.g. < 0.05,
    # while a closed indoor room would have >0.95, and a room with some holes might be in the 0.1-0.8 range.
    # if the ray efficiency is low for an indoor environment, it indicates a lot of ray leak from holes
    # you should repair the mesh in this case for more accurate acoustic rendering
    print('ray efficiency', audio_sensor.getRayEfficiency())

    # plot the waveform of IR and show the audio
    sr = 32000
    samples_clip = 10000
    plt.title('Impulse Response')
    for i in range(2):
        plt.plot(np.linspace(0, samples_clip / sr, samples_clip), ir[i, :samples_clip])
        plt.savefig(
            os.path.join('/content/drive/MyDrive/speech_navigation_output', scene_id, setup_id, 'ir{}'.format(i)))

    plt.clf()

    plt.title('Impulse Response Precomputed')
    for i in range(2):
        plt.plot(np.linspace(0, samples_clip / sr, samples_clip), ir_precomputed[i, :samples_clip])
        plt.savefig(os.path.join('/content/drive/MyDrive/speech_navigation_output', scene_id, setup_id,
                                 'ir_precomputed{}'.format(i)))

    plt.clf()
    if ir_type == 'ir_precomputed':
        ir = ir_precomputed

    # convert input audio from stereo to mono
    sound = AudioSegment.from_wav('/content/drive/MyDrive/speech_navigation/come_here_10s.wav')
    #sound = AudioSegment.from_wav('/content/drive/MyDrive/data_and_checkpoints/eval/audio/YSObG51xihGc.wav')
    #sound = AudioSegment.from_wav('/content/sound-spaces/res/singing.wav')
    #sound = AudioSegment.from_wav('/content/drive/MyDrive/speech_navigation/follow_me.wav')
    sound = sound.set_channels(1)
    sound.export('/content/come_here_10s.wav', format='wav')
    #sound.export("/content/singing.wav", format="wav")
    #sound.export("/content/follow_me_mono.wav", format="wav")
    
    sr, vocal = wavfile.read('/content/come_here_10s.wav')
    #sr, vocal = wavfile.read('/content/singing.wav')
    #sr, vocal = wavfile.read('/content/follow_me_mono.wav')
    print('sr', sr, 'vocal shape', vocal.shape)

    plt.title('Original Speech')
    plt.plot(np.linspace(0, len(vocal) / sr, len(vocal)), vocal)
    # plt.savefig(os.path.join('/content', 'output', 'original_speech'))
    plt.clf()

    # convolve the vocal with IR
    convolved_vocal = np.array([fftconvolve(vocal, ir_channel) for ir_channel in ir])
    print('convolved_vocal.shape', convolved_vocal.shape)
    plt.title('Speech convolved with IR')
    for i in range(2):
        plt.plot(np.linspace(0, len(convolved_vocal[i]) / sr, len(convolved_vocal[i])), convolved_vocal[i])
        plt.savefig(os.path.join('/content/drive/MyDrive/speech_navigation_output', scene_id, setup_id, ir_type,
                                 'speech_convolved_with_ir{}'.format(i)))
    plt.clf()

    # convolved_vocal = convolved_vocal / np.expand_dims(np.max(convolved_vocal, axis=1), axis=1)
    iinfo = np.iinfo(np.int16)
    convolved_vocal = (convolved_vocal / np.max(np.abs(convolved_vocal)) * iinfo.max).astype(np.int16)
    wavfile.write(
        os.path.join('/content/drive/MyDrive/speech_navigation_output', scene_id, setup_id, ir_type, 'reverberent_speech.wav'),
        sr, convolved_vocal.T)

    rt60 = measure_rt60(ir[0], sr, decay_db=30, plot=True)
    print(f'RT60 of the rendered IR is {rt60:.4f} seconds')

    action_names = list(cfg.agents[0].action_space.keys())
    print('action names', action_names)

    dist_pred, elevation_pred, azimuth_pred, dist_gt, azimuth_gt, elevation_gt = run_spatial_ast(ir, scene_id, setup_id)
    azimuth_error = min(abs(azimuth_gt - azimuth_pred), 360 - abs(azimuth_gt - azimuth_pred))
    dist_error = abs(dist_gt - dist_pred)
    if azimuth_error <= 20:
        azimuth_in_range += 1
    if dist_error <= 1:
        dist_in_range += 1
    azimuth_error_sum += azimuth_error
    dist_error_sum += dist_error
    print('azimuth_in_range', azimuth_in_range)
    print('dist_in_range', dist_in_range)
    
    print('total count', total_cnt)

    agent_pos = agent.get_state().position
    goal_pos = dist_angle_to_cartesian(agent_pos, dist_pred, elevation_pred, azimuth_pred)
    print('Goal position', goal_pos)

    print('Finding shortest geodesic path...')
    path_points = get_shortest_path(sim, agent, goal_pos)

    meters_per_pixel = 0.01  # @param {type:"slider", min:0.01, max:1.0, step:0.01}
    # @markdown Customize the map slice height (global y coordinate):
    custom_height = False  # @param {type:"boolean"}
    height = 1

    bounds = sim.pathfinder.get_bounds()
    print("The NavMesh bounds are: " + str(bounds))
    if not custom_height:
        # get bounding box minimum elevation for automatic height
        height = sim.pathfinder.get_bounds()[0][1]

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
    if euclidean_dist(agent_pos, sim.pathfinder.snap_point(source_pos)) <= 1:
        success_cnt += 1
    print('success_cnt', success_cnt)
    
    all_points = path_points + [agent_pos, goal_pos, source_pos]
    pos_pixels = convert_points_to_topdown_pixel_coordinates(sim.pathfinder, all_points, meters_per_pixel)
    print('source position', source_pos)
    print('agent position', agent_pos)
    print('agent rotation quaternion', agent_rot_quat)
    rot_axis, rot_angle = quaternion_to_axis_angle(agent_rot_quat)
    print('rot_axis', rot_axis)
    print('rot_angle degrees ', math.degrees(rot_angle))
    print('source and agent pixel positions', pos_pixels)

    # print("Displaying the raw map from get_topdown_view:")
    # display_map(sim_topdown_map, 'habitat_sim_get_topdown_view', scene_id, setup_id, source_pos=pos_pixels[-1], goal_pos=pos_pixels[-2], agent_pos=pos_pixels[-3], agent_angle=rot_angle, path_points=pos_pixels[:-3])
    print("Displaying the map from the Habitat-Lab maps module:")
    display_map(hablab_topdown_map, 'habitat_lab_get_topdown_map', scene_id, setup_id, source_pos=pos_pixels[-1],
                goal_pos=pos_pixels[-2], agent_pos=pos_pixels[-3], agent_angle=rot_angle, path_points=pos_pixels[:-3])

    # except:
    #    print('error loading config', scene_config['fname'])

azimuth_error_avg = azimuth_error_sum / total_cnt
dist_error_avg = dist_error_sum / total_cnt
success_rate = success_cnt / total_cnt
azimuth_success_rate = azimuth_in_range / total_cnt
dist_success_rate = dist_in_range / total_cnt
print('azimuth_error_avg', azimuth_error_avg)
print('dist_error_avg', dist_error_avg)
print('azimuth_success_rate', azimuth_success_rate)
print('dist_success_rate', dist_success_rate)
print('success_rate', success_rate)

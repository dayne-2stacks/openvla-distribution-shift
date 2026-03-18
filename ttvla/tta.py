from evo_vlac import GAC_model
from evo_vlac.utils.video_tool import compress_video
import os
import subprocess
import time
import redis
import json
#Example code for inputting images and evaluating pair-wise

from transformers import AutoModel




def get_progress(obs, task_description, Critic):
    # print(f"[TTA] get_progress with images={obs}", flush=True)

    ref_images=None
    
    # generate Critic results
    critic_list, value_list=Critic.get_trajectory_critic(
        task=task_description,
        image_list=obs,
        ref_image_list=ref_images,
        batch_num=5,#max batch number when generating critic
        ref_num=0,#image number used in ref_images
        rich=False,#whether to output decimal value
        reverse_eval=False,#whether to reverse the evaluation(for VROC evaluation)
    )


    # print("=" * 100)
    # print(">>>>>>>>>Critic results<<<<<<<<<<")
    # print(" ")

    # print("value_list:")
    # print(value_list)
    # print("=" * 50)

    # print("critic_list:")
    # print(critic_list)
    # print("=" * 50)
    value_list = [float(value) / 100 for value in value_list]
    return value_list



def reward(old_prob, new_prob):
    reward = new_prob - old_prob
    return reward


def start_server(socket_path):
    
    # remove old socket if it exists
    if os.path.exists(socket_path):
        os.remove(socket_path)


    proc = subprocess.Popen([
        "conda", "run", "--no-capture-output", "-n", "ttvla",
        "redis-server",
        "--port", "0",
        "--unixsocket", socket_path,
        "--unixsocketperm", "777"
    ])

    # wait for redis to boot
    time.sleep(1)
    

    print("Redis started")

    return proc

if __name__ == "__main__":
    # model_path = "InternRobotics/VLAC-8b"
    model_path = "/general/dayneguy/cache/modelscope/hub/models/InternRobotics/VLAC-8b"
    socket_path = f"{os.environ['DATA_DIR']}/tmp/redis.sock"
    ready_token = os.environ.get("TTA_READY_TOKEN", "1")
    print(f"[TTA] starting worker", flush=True)
    print(f"[TTA] model_path={model_path}", flush=True)
    print(f"[TTA] socket_path={socket_path}", flush=True)

    proc = start_server(socket_path)
    print("[TTA] redis process launched", flush=True)

    r = redis.Redis(unix_socket_path=socket_path)
    print("[TTA] redis client connected", flush=True)


    print("[TTA] initializing critic...", flush=True)
    Critic = GAC_model(tag='critic')
    Critic.init_model(model_path=model_path, model_type='internvl2', device_map='cuda:0')
    Critic.temperature = 0.5
    Critic.top_k = 1
    Critic.set_config()
    Critic.set_system_prompt()
    print("[TTA] critic initialized", flush=True)

    print("[TTA] running warmup inference...", flush=True)
    while True:
        try:
            test_images=['/data/dayneguy/vla/ttvla/VLAC/evo_vlac/examples/images/test/595-44-565-0.jpg','/data/dayneguy/vla/ttvla/VLAC/evo_vlac/examples/images/test/595-44-565-0.jpg']
            Critic.get_trajectory_critic(
                task="warmup",
                image_list=test_images,
                ref_image_list=None,
                batch_num=5,
                ref_num=0,
                rich=False,
                reverse_eval=False,
            )
            r.set("tta:ready", ready_token)
            # print(f"[TTA] ready flag set after warmup with token {ready_token}", flush=True)
            break
        except Exception as e:
            print(f"[TTA] warmup failed: {e}", flush=True)
            time.sleep(1)

    while True:
        # print("[TTA] waiting for job on tta_images", flush=True)
        _, raw = r.blpop("tta_images")
        # print("[TTA] received raw job", flush=True)

        job = json.loads(raw)
        obs = job["obs"]
        task_description = job["task_description"]
        # print(f"[TTA] old_obs={old_obs}", flush=True)
        # print(f"[TTA] new_obs={new_obs}", flush=True)
        # print(f"[TTA] task={task_description}", flush=True)

        try:
            start = time.time()
            value_list = get_progress(obs, task_description, Critic)
            elapsed = time.time() - start
            # print(f"[TTA] value finished in {elapsed:.2f}s", flush=True)
            print(f"[TTA] value_list={value_list}", flush=True)

            r.rpush("tta_results", json.dumps({"value_list": value_list}))
            # print("[TTA] pushed result to tta_results", flush=True)
        except Exception as e:
            print(f"[TTA] error while processing job: {e}", flush=True)
            r.rpush("tta_results", json.dumps({"value_list": [f'Error: {str(e)}']}))



    

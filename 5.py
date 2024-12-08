import gradio as gr
import cv2
from PIL import Image
import soundfile as sf
import requests
from gradio_client import Client, handle_file
import shutil
from moviepy import VideoFileClip, AudioFileClip

def swapface(image, audio, text, border_image, progress=gr.Progress()):
    progress(0.0)
    # 将numpy数组转换为PIL图像对象
    image_pil = Image.fromarray(image)
    
    # 图片储存在当前目录下的input文件夹中
    image_pil.save("./input/image.jpg", 'JPEG') 

    # 音频储存在当前目录下的input文件夹中
    # sound是一个元组，我们需要将其保存为MP3文件
    sample_rate, audio_data = audio
    sf.write("./input/audio.wav", audio_data, sample_rate)

    url = "http://localhost:5678/swapface"
    data = {'source': 'D:/work3/gradio/input/image.jpg'}

    response = requests.post(url, json=data)
    # 检查响应状态码
    if response.status_code == 200:
        print("请求成功")
        with open('./output/output.mp4', 'wb') as f:
            f.write(response.content)
    else:
        print("请求失败，状态码：", response.status_code)

    progress(0.25)

    #转换语音

    client = Client("http://localhost:9872/")
    result = client.predict(
            ref_wav_path=handle_file('D:/work3/gradio/input/audio.wav'),
            prompt_text=text,
            prompt_language="中文",
            text="看着浩瀚的天空，我们心里洋溢着无限的幸福。",
            text_language="中文",
            how_to_cut="凑四句一切",
            top_k=15,
            top_p=1,
            temperature=1,
            ref_free=False,
            speed=1,
            if_freeze=False,
            inp_refs=[],
            api_name="/get_tts_wav"
    )
    print(result)

    # 源文件路径
    src = result
    # 目标文件路径
    dst = 'D:/work3/gradio/output/output_voice.wav'

    # 复制文件
    shutil.copy(src, dst)

    progress(0.5)

    #转换唇形
    url = "http://localhost:5678/lip"
    data = {'source': dst, 'target': 'D:/work3/gradio/output/output.mp4'}

    response = requests.post(url, json=data)
    # 检查响应状态码
    if response.status_code == 200:
        print("请求成功")
        with open('./output/final_output.mp4', 'wb') as f:
            f.write(response.content)
    else:
        print("请求失败，状态码：", response.status_code)

    progress(0.75)

    #添加边框
    if border_image == "./frame/0.png":
        video = "./output/final_output.mp4"
        return video

    # 读取视频
    cap = cv2.VideoCapture("./output/final_output.mp4")

    # 获取视频的属性
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 读取边框图片
    print(border_image)
    border = cv2.imread(border_image, cv2.IMREAD_UNCHANGED)

    # 检查边框图片是否与视频帧大小相同
    if border.shape[1] != frame_width or border.shape[0] != frame_height:
        raise ValueError("边框图片的尺寸必须与视频帧的尺寸相同")
    
    # 去掉border的alpha通道
    border_rgb = border[:, :, :3]
    
    # 创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编码器
    out = cv2.VideoWriter("./output/temp_output.mp4", fourcc, fps, (frame_width, frame_height))

    # 逐帧处理视频
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 将边框图片覆盖到视频帧上
        # print(frame.shape)
        # print(border.shape)
        # 给frame添加alpha通道
        # frame_alpha = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
        combined_frame = cv2.addWeighted(frame, 1, border_rgb, 1, 0)

        # 写入新视频
        out.write(combined_frame)

        # 显示处理后的帧
        cv2.imshow('Frame', combined_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # 使用moviepy将音频添加回视频
    video_clip = VideoFileClip("./output/temp_output.mp4")
    original_audio = AudioFileClip("./output/final_output.mp4")
    final_clip = video_clip.with_audio(original_audio)
    final_clip.write_videofile("./output/final_output_with_border.mp4", codec='libx264')
    
    video = "./output/final_output_with_border.mp4"
    return video

# gr.Interface(fn=swapface, inputs=[gr.Image(label="输入图片"), gr.Audio(sources="upload", label="请上传3~10秒内参考音频，超过会报错！"), gr.Textbox(label="输入音频文本")],
#               outputs=[gr.Video(label="处理后视频")],allow_flagging="never").launch(server_port=8000), 

def show_image(image_path):
    return image_path

# 预选好的图片路径列表
preselected_images = [
    "./frame/0.png",
    "./frame/1.png",
    "./frame/2.png"
]

# 创建Gradio界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="输入图片")
            audio_input = gr.Audio(sources="upload", label="请上传3~10秒内参考音频，超过会报错！")
            text_input = gr.Textbox(label="输入音频文本")
        with gr.Column():
            border_image = gr.Radio(preselected_images, label="选择边框", type="value", value="./frame/0.png")
            border_show = gr.Image(label="边框预览", value="./frame/0.png")
        output_video = gr.Video(label="处理后视频")

    submit_button = gr.Button("提交")
    submit_button.click(fn=swapface, inputs=[image_input, audio_input, text_input, border_image], outputs=output_video)

    clear_button = gr.Button("清空")
    clear_button.click(fn=lambda: [None, None, None, None, "./frame/0.png"], inputs=[], outputs=[image_input, audio_input, text_input, output_video, border_image])

    border_image.change(fn=show_image, inputs=border_image, outputs=border_show)
demo.launch(server_port=8000)
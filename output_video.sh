#https://raspberrypi.stackexchange.com/questions/23182/how-to-stream-video-from-raspberry-pi-camera-and-watch-it-live
#There are several options you can choose between. At my work we are using VLC to stream video captured by Raspberry Pi Camera from our server-rooms to the office. One downside of this is that there are about 5 seconds delay and I haven't found a solution to this. The following is our setup:

raspivid -o - -t 0 -hf -w 640 -h 360 -fps 25 | cvlc -vvv stream:///dev/video0 --sout '#rtp{sdp=rtsp://:8554}' :demux=h264
#sudo modprobe bcm2835-v4l2
#cvlc v4l2:///dev/video0 --v4l2-width 1920 --v4l2-height 1080 --v4l2-chroma h264 --sout '#standard{access=http,mux=ts,dst=0.0.0.0:8554}'

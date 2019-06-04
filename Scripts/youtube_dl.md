# youtube-dl to download audio from youtube videos.
sudo curl -L https://yt-dl.org/downloads/latest/youtube-dl -o /usr/local/bin/youtube-dl
sudo chmod a+rx /usr/local/bin/youtube-dl

youtube-dl --extract-audio --audio-format wav https://www.youtube.com/watch?v=0eO1-lqmv7E

# Use SoX to trim audio file
sudo apt-get install sox

sox input.file output.file trim 10 	# Cuts off frist 10 seconds
sox input.file output.file trim 0 X # Cuts clip from 0 seconds until X seconds.


from moviepy import VideoFileClip, AudioFileClip, afx
import os

class VideoMuxer:
    def __init__(self):
        pass

    def mux_audio_video(self, video_path, audio_path, output_path="final_output.mp4"):
        """
        Takes the original video and loops the generated audio over its duration.
        """
        print(f"Muxing {audio_path} into {video_path}...")
        
        if not os.path.exists(video_path):
            return {"error": f"Original video not found: {video_path}"}
        if not os.path.exists(audio_path):
            return {"error": f"Audio file not found: {audio_path}"}
            
        try:
            video_clip = VideoFileClip(video_path)
            audio_clip = AudioFileClip(audio_path)
            
            # Our generated music track will likely be shorter than the video.
            # We loop the generated soundtrack to precisely cover the overall video length.
            looped_audio = audio_clip.with_effects([afx.AudioLoop(duration=video_clip.duration)])
            
            # We attach the new background track to our original video
            final_video = video_clip.with_audio(looped_audio)
            
            # Write final export
            final_video.write_videofile(
                output_path, 
                codec="libx264", 
                audio_codec="aac",
                logger=None # Prevent printing 100 progress bars
            )
            
            video_clip.close()
            audio_clip.close()
            final_video.close()
            
            print(f"Successfully generated final output: {output_path}")
            return output_path
            
        except Exception as e:
            return {"error": f"Failed to mux video and audio: {e}"}

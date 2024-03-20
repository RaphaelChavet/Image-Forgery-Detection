import os
import cv2
import argparse

from tqdm import tqdm

def main(path_to_directory_with_videos):

	#breakpoint()
	for video_name in tqdm(os.listdir(path_to_directory_with_videos)):
		if video_name.endswith(".avi") or video_name.endswith(".3gpp") or video_name.endswith(".mp4"):
			full_video_dir_path = os.path.join(path_to_directory_with_videos, video_name[:-4])

			if not os.path.exists(full_video_dir_path):
				os.mkdir(full_video_dir_path)

			frame_counter = 0
			cap = cv2.VideoCapture(os.path.join(path_to_directory_with_videos, video_name))
			length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
			while cap.isOpened():
				ret, frame = cap.read()

				# if frame is read correctly ret is True
				if not ret:
					if frame_counter != length:
						print("Not all frames have been read!")

					break
				
				frame_name = video_name[:-4] + "_frame_" + str(frame_counter) + ".jpg"
				frame_full_path = os.path.join(full_video_dir_path, frame_name)
				cv2.imwrite(frame_full_path, frame)

				frame_counter += 1

			cap.release()
			cv2.destroyAllWindows()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Define the parameters')
	
	parser.add_argument('--path_to_directory_with_videos', type=str, default="/hadatasets/gabriel_bertocco/ForensicsDatasets/Training/Forged/Forgery_deletion", help='Path to the directories where the videos are')
	
	args = parser.parse_args()

	path_to_directory_with_videos = args.path_to_directory_with_videos
	
	main(path_to_directory_with_videos)
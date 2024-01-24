from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
import yt_dlp as youtube_dl
import json
import re
import os
from typing import List, Dict, Union

def download_video_transcript(video_id: str, yt_video_path: str) -> Union[Dict, str]:

    try:
        # Download the transcript using youtube-transcript-api
        raw_transcript = YouTubeTranscriptApi.get_transcript(video_id, preserve_formatting=True, languages=['en-US'])

        return raw_transcript
    
    except NoTranscriptFound:
        # Handle the case where no transcript is found
        return None
    

def get_video_meta_info(yt_video_path: str) -> Dict:

    # youtube-dl options
    ydl_opts = {
        'quiet': False,
        'skip_download': True,
        'embed-chapters': True
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        video_info = ydl.extract_info(yt_video_path, download=False)

        return {
            'title': video_info.get('title'),
            'duration': video_info.get('duration'),
            'view_count': video_info.get('view_count'),
            'like_count': video_info.get('like_count', 'Not available'),
            'chapters': video_info.get('chapters', [])
        }


def save_to_json(video_metadata: Dict, output_file: str) -> None:

    with open(output_file, 'w') as file:
        json.dump(video_metadata, file, indent=4)


def save_to_txt(text: str, output_file: str) -> None:

    with open(output_file, 'w') as file:
        file.write(text)

def format_title(title: str) -> str:
    # Replace non-word characters with an underscore
    formatted_title = re.sub(r'\W+', '_', title)

    # Remove potential leading and trailing underscores
    formatted_title = formatted_title.strip('_')

    return formatted_title


def prettify_transcript(chapterwise_transcript: Dict) -> str:

    transcript = ""
    for chapter_title, chapter_text in chapterwise_transcript.items():
        transcript += "**"*50
        transcript += "\n"
        transcript += f"CHAPTER: {chapter_title}"
        transcript += "\n\n"
        transcript += chapter_text
        transcript += "\n\n"
    
    return transcript

def split_transcript_by_chapters(transcript: List[Dict], chapters: List[Dict]) -> Dict:
    
    chapterwise_transcript = {chapter['title']: [] for chapter in chapters}

    for caption in transcript:
        for chapter in chapters:
            if chapter['start_time'] <= caption['start'] <= chapter['end_time']:

                # Add the caption to the corresponding chapter
                chapterwise_transcript[chapter['title']].append(caption['text'])
                break  # Break the loop once the correct chapter is found
    
    
    for chapter_title, chapter_text in chapterwise_transcript.items():
        transcript_text = ' '.join([caption.replace("\n", " ") for caption in chapter_text])
        transcript_text = transcript_text.replace(". ", ".\n")

        chapterwise_transcript[chapter_title] = transcript_text
    
    return chapterwise_transcript


def prepare_video_data(video_id: str, parent_folder: str) -> None:
    yt_video_path = f"https://www.youtube.com/watch?v={video_id}"
    print (f"Processing {yt_video_path}...")

    raw_transcript = download_video_transcript(video_id=video_id, yt_video_path=yt_video_path)

    if raw_transcript is None:
        print ("No transcript found for this video. So skipping the processing.")
        return


    video_metadata = get_video_meta_info(yt_video_path=yt_video_path)

    chapterwise_transcript = split_transcript_by_chapters(transcript=raw_transcript, chapters=video_metadata['chapters'])

    pretty_transcript = prettify_transcript(chapterwise_transcript)

    video_metadata['video_id'] = video_id
    video_metadata['video_url'] = yt_video_path

    video_folder_name = format_title(video_metadata['title'])
    video_folder_name = os.path.join(parent_folder, video_folder_name)

    if not os.path.exists(video_folder_name):
        os.makedirs(video_folder_name)
    
    metadata_path = os.path.join(video_folder_name, "metadata.json")
    transcript_path = os.path.join(video_folder_name, "transcript.txt")

    print (f"Saved metadata for video with ID {video_id} in {metadata_path}")
    save_to_json(video_metadata=video_metadata, output_file=metadata_path)

    print (f"Saved transcripts for video with ID {video_id} in {transcript_path}")
    save_to_txt(text=pretty_transcript, output_file=transcript_path)




def get_relevant_videos_from_playlist(playlist_url):
    ydl_opts = {
        'quiet': True,
        'extract_flat': True,
        'force_generic_extractor': True,
    }

    relevant_videos = []
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(playlist_url, download=False)
        if 'entries' in info_dict:
        
            for video in info_dict['entries']:
                if is_relevant(video_title=video.get('title'), video_start_idx = 299, video_end_idx = 358):
                    relevant_videos.append(
                        {
                            'video_title': video.get('title'),
                            'video_url': video.get('url'),
                            'video_id': video.get('id')
                        }
                    )

    return relevant_videos


def is_relevant(video_title, video_start_idx, video_end_idx):
    return (match := re.search(r'#(\d+)$', video_title)) and video_start_idx <= int(match.group(1)) <= video_end_idx



if __name__ == "__main__":

    # create parent folder called 'transcripts'
    root_transcript_folder = 'lex_transcripts'

    if not os.path.exists(root_transcript_folder):
        os.makedirs(root_transcript_folder)
    
    # get all video ids of the Lex Fridman Podcast playlist
    playlist_url = 'https://www.youtube.com/playlist?list=PLrAXtmErZgOdP_8GztsuKi9nrraNbKKp4'
    relevant_video_titles = get_relevant_videos_from_playlist(playlist_url)

    # iterate over them and create the metadata and transcript
    for video in relevant_video_titles:
        prepare_video_data(video_id=video['video_id'], parent_folder=root_transcript_folder)



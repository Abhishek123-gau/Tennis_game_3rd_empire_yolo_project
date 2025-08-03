
from ultralytics import YOLO
import cv2
import pickle
import pandas as pd
class BallTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)
    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # interpolate the missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions
    # It's going to return all the bbox under every frame so nti is kind of array of dictionzry
    def get_ball_shot_frames(self,ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        df_ball_positions['ball_hit'] = 0

        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2'])/2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()
        minimum_change_frames_for_hit = 25
        for i in range(1,len(df_ball_positions)- int(minimum_change_frames_for_hit*1.2) ):
            negative_position_change = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[i+1] <0
            positive_position_change = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[i+1] >0

            if negative_position_change or positive_position_change:
                change_count = 0 
                for change_frame in range(i+1, i+int(minimum_change_frames_for_hit*1.2)+1):
                    negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[change_frame] <0
                    positive_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[change_frame] >0

                    if negative_position_change and negative_position_change_following_frame:
                        change_count+=1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count+=1
            
                if change_count>minimum_change_frames_for_hit-1:
                    df_ball_positions['ball_hit'].iloc[i] = 1

        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit']==1].index.tolist()

        return frame_nums_with_ball_hits
    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        ball_detection = []
        if(read_from_stub and stub_path!=None):
            with open(stub_path,'rb') as f:
                player_detections = pickle.load(f)
            return player_detections
        if(stub_path!=None):
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detection, f)
        for frame in frames:     
            player_dict=self.detect_frame(frame)
            ball_detection.append(player_dict)
        return ball_detection

    # This wil return the dictionary
    def detect_frame(self,frame):
        results=self.model.track(frame, conf=0.15,persist=True)[0] #persist is toi remember the pervious tracking
        # id_name_dict=results.names
        ball_dict = {}
        for box in results.boxes:
            # track_id = int(box.id.tolist()[0])
            result= box.xyxy.tolist()[0]
            # objext_cls_id = box.cls.tolist()[0] 
            # object_cls_name=id_name_dict[objext_cls_id]
            
            ball_dict[1]=result
        return ball_dict
    # now let's concate these bbox on the images

    def draw_bboxes(self,video_frame,player_detection):
        output_video_frames=[]
        for frame,player_dict in zip(video_frame,player_detection):
            for track_id, bbox in player_dict.items():
                x1,y1, x2,y2 = bbox
                cv2.putText(frame,f"ball_id : {track_id}",(int(bbox[0]),int(bbox[1]-10)),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,225),2)

                cv2.rectangle(frame,(int(x1), int(y1)) ,(int(x2), int(y2)),(255,0,255),2)
            output_video_frames.append(frame)
        return output_video_frames

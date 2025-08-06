from annotations import *
def load_image(subdir,clip, frame_id):
    
    path = os.path.join("/kaggle/input/volleyball/volleyball_/videos",subdir ,clip, frame_id)
    img = cv2.imread(path)
    return img


def display_image(subdir, clip, frame_id):
    path = os.path.join("/kaggle/input/volleyball/volleyball_/videos", subdir, "annotations.txt")
    video_info = read_file(path)
    
    groupactivity = video_info[frame_id]["groupactivity"]
    boxinfos = video_info[frame_id]["boxinfos"]

    img = load_image(subdir, clip, frame_id)
    
    if img is None:
        print(f"Image not found at path: /kaggle/input/volleyball/volleyball_/videos/{subdir}/{clip}/{frame_id}")
        return

    color = (255, 0, 0)  # Red boxes
    thickness = 2

    for box in boxinfos:
        start_point = (box.x1, box.y1)
        end_point = (box.x2, box.y2)
        cv2.rectangle(img, start_point, end_point, color, thickness)

    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.imshow(img_rgb)
    plt.title(f"Group Activity: {groupactivity}")
    plt.axis('off')
    plt.show()




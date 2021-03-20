import argparse
import sys
import os
from utils import *

#terminal input: python3 yoloface.py --folder samples/ --output-dir outputs/

parser = argparse.ArgumentParser()
parser.add_argument('--model-cfg', type=str, default='./cfg/yolov3-face.cfg',
                    help='path to config file')
parser.add_argument('--model-weights', type=str,
                    default='./model-weights/yolov3-wider_16000.weights',
                    help='path to weights of model')
parser.add_argument('--folder', type=str, default='',
                    help='path to folder')
parser.add_argument('--output-dir', type=str, default='outputs/',
                    help='path to the output directory')
args = parser.parse_args()

# check outputs directory
if not os.path.exists(args.output_dir):
    print('==> Creating the {} directory...'.format(args.output_dir))
    os.makedirs(args.output_dir)
else:
    print('==> Skipping create the {} directory...'.format(args.output_dir))

# Give the configuration and weight files for the model and load the network
# using them.
net = cv2.dnn.readNetFromDarknet(args.model_cfg, args.model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def detectFaces():

    faces_detected = 0
    input_files = []
    output_file = ''

    if args.folder:
        #Error if no folder if found.
        if not os.path.isdir(args.folder):
            print("[!] ==> Input directory {} doesn't exist".format(args.folder))
            sys.exit(1)
        
        for filename in os.listdir(args.folder):
            cap = cv2.VideoCapture(os.path.join(args.folder, filename))
            output_file = filename[:-4].rsplit('/')[-1] + '_yoloface.jpg'

            while True:
                has_frame, frame = cap.read()
 
                if not has_frame:
                    cv2.waitKey(1000)
                    break

                # Create a 4D blob from a frame.
                blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                            [0, 0, 0], 1, crop=False)

                # Sets the input to the network
                net.setInput(blob)

                # Runs the forward pass to get output of the output layers
                outs = net.forward(get_outputs_names(net))

                # Remove the bounding boxes with low confidence
                faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
                print('[i] ==> # detected faces: {}'.format(len(faces)))
                faces_detected += len(faces)

                # initialize the set of information we'll displaying on the frame
                info = [
                    ('number of faces detected', '{}'.format(len(faces)))
                ]

                for (i, (txt, val)) in enumerate(info):
                    text = '{}: {}'.format(txt, val)
                    cv2.putText(frame, text, (10, (i * 20) + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)

                # Save the output video to file
                if args.folder:
                    cv2.imwrite(os.path.join(args.output_dir, output_file), frame.astype(np.uint8))

                key = cv2.waitKey(1)
                if key == 27 or key == ord('q'):
                    print('[i] ==> Interrupted by user!')
                    break

            cap.release()
            cv2.destroyAllWindows()
    return faces_detected

def actualFaces():
    path_xml = '/home/xdapperdonx/Desktop/Code/Undergraduate/Dataset/annotations'
    xmls = []
    counter = 0

    for file_xml in os.listdir(path_xml):
        if file_xml.endswith('.xml'):
            xmls.append(file_xml)

    for data in xmls:
        
        tree = ET.ElementTree(file=os.path.join(path_xml,data))
        root = tree.getroot()

        for elem in root.findall('./object/name'):
            if elem.text == "with_mask" or elem.text == "without_mask":
                counter += 1
    return counter

def calculation(faces_detected, actual_faces):
    print((faces_detected/actual_faces) * 100)


if __name__ == '__main__':
    calculation(detectFaces(), actualFaces())
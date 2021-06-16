from tkinter import Image
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from keras_vggface.utils import preprocess_input
from keras import backend as K
from keras.applications.resnet50 import decode_predictions
from keras_applications.resnet50 import preprocess_input
from matplotlib import pyplot
K.set_image_data_format('channels_first')
from mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
import cv2


"""
Retreive frame using a camera
"""


def image_capture():
    # define a video capture object
    vid = cv2.VideoCapture(0)
    while True:
        # Capture the video frame
        ret, frame = vid.read()

        # Save the resulting frame
        cv2.imwrite('opencv' + str(i) + '.png', frame)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()



"""
extract a single face from a given photograph
"""


def extract_face(filename, required_size=(224, 224)):
    # load image from file
    pixels = pyplot.imread(filename)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array

"""
extract faces and calculate face embeddings for a list of photo files
"""

def get_embeddings(filenames):
    # extract faces
    faces = [extract_face(f) for f in filenames]
    # convert into an array of samples
    samples = asarray(faces, 'float32')
    # prepare the face for the model, e.g. center pixels
    samples = preprocess_input(samples, version=2)
    # create a vggface model
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    # perform prediction
    yhat = model.predict(samples)
    return yhat


"""
determine if a candidate face is a match for a known face
"""


def is_match(known_embedding, candidate_embedding, thresh=0.5):
    # calculate distance between embeddings
    score = cosine(known_embedding, candidate_embedding)
    if score <= thresh:
        print('>face is a Match (%.3f <= %.3f)' % (score, thresh))
    else:
        print('>face is NOT a Match (%.3f > %.3f)' % (score, thresh))


"""
define filenames
"""
filenames =[]
for i in range(1, 100):
    filenames.append('opencv' + str(i) + '.png')

"""
Get embeddings & verify face
"""
# get embeddings file filenames
embeddings = get_embeddings(filenames)
# define an image to verify
name_id = embeddings[0]
# verify known photos of name
for i in range(1,100):
    is_match(embeddings[0], embeddings[i])

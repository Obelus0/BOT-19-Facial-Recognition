# VGGFace-Implementation
  Implemented Facial Verification using VGGFace.
  
  Here VGGFace is provides a face embedding, which is compared in order to provide verification.
  
Training of VGGFace is such that the Euclidean distance between vectors generated for the same identity are made smaller and the vectors generated for different identities is made larger. This is achieved using a triplet loss function.

# References 
https://github.com/rcmalli/keras-vggface

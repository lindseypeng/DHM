# DHM
convolution method with automation of finding multiple focused planes for objects in python
in progress and development

the first part is the reconstruction of intensity images from a single hologram
the second part is finding focused planes 

To find the focused planes, there are two major steps

1.locate x-y locations of objects
a)project all the reconstructed planes in spatial domain onto one 2-d minimum intensity projection 
(since the object obsorbed most of the incoming light)
b)using sauvola thresholding,binarize the image into background and foreground
(this is good for document images, such as for uneven light illumination, many papers uses otsu 
threshold but performs bad for uneven illuminations. a survey has showed sauvola to be 2nd best method for document images
among 46 algorithms)
c)using contours and image moments to find needed information such as centroid, and all the pixels within the foreground objects

2.locate focused z plane for each object 
a)using the xy objects coordinates, we can use entropy, or minimum intensity, to find the plane that each 
object is in focused.
b)find the average or mode of these z planes and decide which ones are most likely to be focused planes
c)only these planes are transferred back to CPU from GPU if needed to save transferring time

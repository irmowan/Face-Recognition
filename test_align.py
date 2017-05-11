import input, transform, cv2
queue = input.generate_input_queue('casia_100.txt')
for i in xrange(100):
    file_path = queue[0][0]
    image, label, landmark = input.read_image_from_disk(queue)
    im_rot, im_rez, crop, ii = transform.img_process(image, landmark)
    path = 'test/' + file_path[-7:-4] 
    cv2.imwrite(path + '_in.jpg', image)
    cv2.imwrite(path + '_rot.jpg', im_rot)
    cv2.imwrite(path + '_rez.jpg', im_rez)
    cv2.imwrite(path + '_crop.jpg', crop)
    cv2.imwrite(path + '_out.jpg', ii)


from utils import get_and_save_person_face_embeddings, \
    get_and_save_person_face_depth_maps, \
    delete_person_face_embeddings, view_face_embeddings, \
    insert_delete_update_person_face_data_in_database, \
    insert_delete_update_person_face_data_in_database_lan
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect face embeddings and depth maps for face recognition')

    parser.add_argument('--script', type=str, default='1',
    help='Select which script to use (0, 1, 2)')

    parser.add_argument('--n_img', type=int, default=10,
    help='How much images to use per person')

    parser.add_argument('--save_img', type=bool, default=False,
    help='Whether or not to save images of faces')
    parser.add_argument('--save_path', type=str, default="face_img_data",
    help='Save path for images when save images is active')

    parser.add_argument('--save_dm', type=bool, default=False,
    help='Whether or not to save depth maps of faces')

    parser.add_argument('-p', '--port', type=int, default=9999,
    help='Choose a port for a server to listen at, if not provided port 9999 will be used')

    args = parser.parse_args()


    #get_and_save_person_face_embeddings("face_embeddings_data", n_img_class = 10,
    # save_face_depth_maps = True, save_face_img = True, face_img_path = "face_img_data")

    #get_and_save_person_face_depth_maps("face_embeddings_data", n_img_class = 10)

    #delete_person_face_embeddings("face_embeddings_data", "")

    #view_face_embeddings("face_embeddings_data")


    if args.script == '0':
        insert_delete_update_person_face_data_in_database(args.n_img,
        args.save_img, args.save_path)
    elif args.script == '1':
        insert_delete_update_person_face_data_in_database_lan(args.n_img,
        args.port, args.save_dm)
    elif args.script == '2':
        pass
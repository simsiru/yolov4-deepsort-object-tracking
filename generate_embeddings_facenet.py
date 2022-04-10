from utils import get_and_save_person_face_embeddings, get_and_save_person_face_depth_maps, delete_person_face_embeddings, view_face_embeddings


if __name__ == "__main__":

    get_and_save_person_face_embeddings("face_embeddings_data", n_img_class = 10, save_face_depth_maps = True, save_face_img = True, face_img_path = "face_img_data")

    #get_and_save_person_face_depth_maps("face_embeddings_data", n_img_class = 10)

    #delete_person_face_embeddings("face_embeddings_data", "")

    #view_face_embeddings("face_embeddings_data")

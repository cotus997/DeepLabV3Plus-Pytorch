def undo_transformations_numpy(image_tensor, mean, std):
    """
    Inverte le trasformazioni di normalizzazione e converte un tensore in formato numpy per la visualizzazione.

    Args:
        image_tensor (torch.Tensor): Immagine normalizzata (C, H, W).
        mean (list): Media usata per la normalizzazione [R_mean, G_mean, B_mean].
        std (list): Deviazione standard usata per la normalizzazione [R_std, G_std, B_std].

    Returns:
        np.ndarray: Immagine (H, W, C) in formato numpy [0, 255].
    """
    # Convertire il tensore in numpy e permutare i canali da (C, H, W) a (H, W, C)
    image = np.array(image_tensor)
    image = np.transpose(image, (1, 2, 0))
    # image = image_tensor.permute(1, 2, 0).cpu().numpy()

    # Invertire la normalizzazione
    image = image * std + mean

    # Riporta i valori a [0, 255]
    image = np.clip(image * 255, 0, 255).astype(np.uint8)

    return image


# Supponiamo che tu abbia già creato un'istanza del dataset
# dataset = Cityscapes(root='path_to_cityscapes', split='train', transform=None)

# Carica un'immagine e la sua maschera
index = 200  # Scegli un indice a caso
image, road_mask = train_dst[index]

# Converti l'immagine e la maschera in formato numpy per la visualizzazione
# image_np = np.array(image).astype(np.uint8)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
image_np = undo_transformations_numpy(image, mean, std)
road_mask_np = np.array(road_mask).astype(np.uint8) + 10
# Visualizza l'immagine e la maschera
plt.figure(figsize=(10, 5))

# Mostra l'immagine
plt.subplot(1, 2, 1)
plt.imshow(image_np)
plt.title('Image')
plt.axis('off')

# Mostra la maschera della classe "road"
plt.subplot(1, 2, 2)
plt.imshow(road_mask_np, cmap='gray')  # Usa 'gray' per la maschera binaria
plt.title('Road Mask')
plt.axis('off')

plt.show()
exit(0)
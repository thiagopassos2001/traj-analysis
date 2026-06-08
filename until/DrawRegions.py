import cv2
import numpy as np

img_ref = r"C:\Users\thiag\OneDrive\Documents\Repositórios (Local)\visao-comp\4 Runs\img_ref\Faixa Azul (Fortaleza)\Humberto Monte\2025-08-06\DJI_0002_5160.png"

# Lista de pontos no sistema ORIGINAL
points_original = []

# Fator de escala (imagem exibida)
scale = 1.0

# Cópias globais
img_display = None
img_display_draw = None


def draw_polygon():
    """ Desenha o polígono na imagem exibida (redimensionada). """
    global img_display, img_display_draw, points_original, scale

    img_display_draw = img_display.copy()

    if len(points_original) > 0:
        # Desenhar pontos
        for p in points_original:
            x_disp = int(p[0] * scale)
            y_disp = int(p[1] * scale)
            cv2.circle(img_display_draw, (x_disp, y_disp), 3, (0, 0, 255), -1)

        # Desenhar linhas
        for i in range(1, len(points_original)):
            x1, y1 = points_original[i - 1]
            x2, y2 = points_original[i]
            cv2.line(img_display_draw,
                     (int(x1 * scale), int(y1 * scale)),
                     (int(x2 * scale), int(y2 * scale)),
                     (0, 255, 0), 2)

    cv2.imshow("Imagem", img_display_draw)


def click_event(event, x, y, flags, param):
    global points_original, scale

    if event == cv2.EVENT_LBUTTONDOWN:
        # Converter para coordenadas originais
        orig_x = int(x / scale)
        orig_y = int(y / scale)

        points_original.append((orig_x, orig_y))

        draw_polygon()   # redesenha o polígono atualizado


def main(img_ref):
    global img_display, img_display_draw, scale

    img = cv2.imread(img_ref)
    if img is None:
        print("Erro ao carregar imagem.")
        return

    h, w = img.shape[:2]

    # Dimensão máxima para exibição
    # max_w, max_h = 1280, 720

    # Calcular escala
    scale = 0.7 # min(max_w / w, max_h / h, 1.0)
    print(scale)

    # Criar imagem exibida
    img_display = cv2.resize(img, (int(w * scale), int(h * scale)))
    img_display_draw = img_display.copy()

    cv2.imshow("Imagem", img_display)
    cv2.setMouseCallback("Imagem", click_event)

    print("Clique para adicionar pontos ao polígono.")
    print("Pressione ENTER para finalizar.")
    print("Pressione ESC para cancelar.")

    while True:
        key = cv2.waitKey(1) & 0xFF

        # ENTER finaliza
        if key == 13:
            break

        # ESC cancela
        if key == 27:
            print("Cancelado.")
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()

    print("\n✔ Pontos do polígono (coordenadas originais):")
    print(points_original)

    print("\n✔ Formato Shapely:")
    print(f"Polygon({points_original})")


if __name__ == "__main__":
    main(img_ref)

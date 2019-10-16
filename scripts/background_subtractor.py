import cv2 as cv
import numpy as np

from argparse import ArgumentParser


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "-r", "--ratio", default=0.85, type=float, help="Set background ratio"
    )
    ap.add_argument("-s", "--shadow", default=255, type=int, help="Set shadow value")
    ap.add_argument(
        "-rec", "--record", default=False, action="store_true", help="Record output?"
    )
    args = vars(ap.parse_args())

    cap = cv.VideoCapture(0)
    if args["record"]:
        h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        codec = cv.VideoWriter_fourcc(*"MPEG")
        out = cv.VideoWriter("out.avi", codec, 10.0, (w * 2, h))

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (6, 6))

    bgs = cv.createBackgroundSubtractorMOG2()
    bgs.setBackgroundRatio(args["ratio"])
    bgs.setShadowValue(args["shadow"])

    while True:
        grabbed, frame = cap.read()
        if not grabbed:
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (5, 5), 1)
        foreground = bgs.apply(blur)
        closed = cv.morphologyEx(foreground, cv.MORPH_CLOSE, kernel)
        contours = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_L1)[0]
        mask = np.zeros_like(gray)
        if contours is not None:
            if len(contours):
                contour = sorted(contours, reverse=True, key=cv.contourArea)[0]
                rect = cv.minAreaRect(contour)
                box = cv.boxPoints(rect)
                box = np.int0(box)
                cv.drawContours(mask, [box], 0, (255, 255, 255), -1)

            frame_masked = cv.bitwise_and(frame, frame, mask=mask)
            output = np.hstack([frame, frame_masked])
            if args["record"]:
                out.write(output)
            cv.imshow("output", output)
            k = cv.waitKey(1) & 0xFF
            if k == ord("q"):
                break

    cap.release()
    if args["record"]:
        out.release()
    cv.destroyAllWindows()

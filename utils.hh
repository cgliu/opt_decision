#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

Mat flip(Mat img)
{
    Mat out;
    flip(img, out, 0);
    return out;
}

//
// Convert a Mat to a colormap Mat
// @param [in] data Mat
// @return     Colormap Mat
//
Mat for_show(Mat img)
{
    Mat img_sc;
    img.convertTo(img_sc, CV_8U, 255.0);
    Mat img_cm;
    applyColorMap(img_sc, img_cm, COLORMAP_JET);
    return flip(img_cm);
}

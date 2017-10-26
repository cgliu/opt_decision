#pragma once

// #include <opencv2/contrib/contrib.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

cv::Mat flip(Mat img)
{
    cv::Mat out;
    cv::flip(img, out, 0);
    return out;
}

//
// Convert a Mat to a colormap Mat
// @param [in] data Mat
// @return     Colormap Mat
//
cv::Mat for_show(cv::Mat img)
{
    cv::Mat img_sc;
    img.convertTo(img_sc, CV_8U, 255.0);
    cv::Mat img_cm;
    cv::applyColorMap(img_sc, img_cm, COLORMAP_JET);
    return flip(img_cm);
}

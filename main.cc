#include "dynamics.hh"
#include "dp.hh"

#include <array>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <vector>
#include <tuple>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

constexpr double dt = 1.0;
constexpr double horizon_s = 10.0;
constexpr double lookahead_m = 100.0;
constexpr double max_speed = 10.0;
constexpr double min_speed = 0.0;

int main()
{
    const double map_width_s = 10.0;
    const double map_height_m = 100.0;
    // read image
    Mat img = imread("pt_map.png", 0);   // Read the file
    img.convertTo(img, CV_64FC1);
    //
    pt_map map(img, map_width_s, map_height_m);

    vehicle_model car(map);

    DPClass<vehicle_model> dp(&car);

    double step_p = 1;
    double step_v = 0.1;
    double step_t = 1;
    for(double t=map_width_s - step_t; t >= 0.0; t -=step_t)
    {
        for(double p = 0l; p < map_height_m; p+= step_p)
        {
            for(double v = 0.0l; v < max_speed; v += step_v)
            {
                vehicle_model::state x = {p, v};
                dp.update(x, t);
            }
        }
    }

    auto mat = dp.get_all_value();
    normalize(mat, mat, 0, 1, NORM_MINMAX, -1, Mat());

    auto policy_t = dp.get_all_policy();
    normalize(policy_t, policy_t, 0, 1, NORM_MINMAX, -1, Mat());

    //Initialize m
    double minVal;
    double maxVal;
    Point minLoc;
    Point maxLoc;
    minMaxLoc( mat, &minVal, &maxVal, &minLoc, &maxLoc );

    cout << "min val : " << minVal << endl;
    cout << "max val: " << maxVal << endl;

    namedWindow( "Display Value", WINDOW_NORMAL);// Create a window for display.
    imshow( "Display Value", mat);               // Show our image inside it.

    namedWindow( "Display Policy", WINDOW_NORMAL);// Create a window for display.
    imshow( "Display Policy", policy_t);               // Show our image inside it.

    waitKey(0);                                   // Wait for a keystroke in the window

    return 0;
}

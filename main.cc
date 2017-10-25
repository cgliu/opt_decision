#include "dynamics.hh"
#include "dp.hh"
#include "utils.hh"

#include <array>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <vector>
#include <tuple>


using namespace std;
using namespace cv;

constexpr double dt = 1.0;
constexpr double horizon_s = 10.0;
constexpr double lookahead_m = 100.0;
constexpr double max_speed = 10.0;
constexpr double min_speed = 0.0;



int main()
{
    using state = vehicle_model::state;
    using control = vehicle_model::control;
    const double map_width_s = 10.0;
    const double map_height_m = 100.0;
    // read image
    Mat img = imread("pt_map.png", 0);   // Read the file
    // convert white to 0, black to 1
    img = flip(img);
    Mat dest;
    threshold(img, dest, 127, 1, CV_THRESH_BINARY_INV);
    dest.convertTo(dest, CV_64FC1);

    pt_map map(dest, map_width_s, map_height_m);

    vehicle_model car(&map);

    double step_p = 1.0;
    double step_v = 1.0;
    double step_t = 0.5;

    DPClass<vehicle_model> dp(&car, step_t, step_p, step_v);

    for(double t=map_width_s - step_t; t >= 0.0; t -=step_t)
    {
        for(double p = 0l; p < map_height_m; p+= step_p)
        {
            for(double v = 0.0l; v < max_speed; v += step_v)
            {
                state x = {p, v};
                dp.update(x, t);
            }
        }
    }

    auto value_t = dp.get_all_value();
    normalize(value_t, value_t, 0, 1, NORM_MINMAX, -1, Mat());

    auto policy_t = dp.get_all_policy();
    normalize(policy_t, policy_t, 0, 1, NORM_MINMAX, -1, Mat());

    // show the pt-map
    namedWindow( "pt-map", WINDOW_NORMAL);
    imshow( "pt-map", flip(img));

    namedWindow( "Display Value", WINDOW_NORMAL);// Create a window for display.
    imshow( "Display Value", for_show(value_t));               // Show our image inside it.

    namedWindow( "Display Policy", WINDOW_NORMAL);// Create a window for display.
    imshow( "Display Policy", for_show(policy_t));               // Show our image inside it.

    // show optimal trajectory on pt_map

    pt_map overlay_map(img * 0.5, map_width_s, map_height_m);

    state x = {20.0, 5.0};
    double sim_dt = 0.5;
    for(double time = 0; time < 10.0; time += sim_dt)
    {
        overlay_map.draw_circle(x[0], time);
        const auto u = dp.policy_get(x, time);
        cout << time << " " << u[0] << endl;
        x = car.dynamics(x, u, sim_dt);
    }

    namedWindow( "Display Trajectory", WINDOW_NORMAL);// Create a window for display.
    imshow( "Display Trajectory", flip(overlay_map.get_value()));               // Show our image inside it.

    waitKey(0);                                   // Wait for a keystroke in the window

    return 0;
}

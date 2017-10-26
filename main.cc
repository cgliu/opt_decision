// Optimal decisoion making using Dynamic Programming

#include "dynamics.hh"
#include "dp.hh"
#include "utils.hh"

#include <array>
#include <chrono>  // for high_resolution_clock
#include <cassert>
#include <iostream>
#include <iomanip>
#include <tuple>
#include <thread>
#include <vector>


using namespace std;
using namespace cv;
using state = vehicle_model::state;
using control = vehicle_model::control;

void task(int n, int id)
{
    this_thread::sleep_for (chrono::seconds(n));
    std::cout << "task" << id << " done." << endl;
}

void value_iteration(size_t job,
                     DPClass<vehicle_model> * dp,
                     double t,
                     double max_p, double min_p, double step_p,
                     double max_v, double min_v, double step_v)
{

    for(double p = min_p; p < max_p; p+= step_p)
    {
        for(double v = min_v; v < max_v; v += step_v)
        {
            state x = {p, v};
            dp->update(x, t);
        }
    }
}

int main()
{

    const double map_width_s = 10.0;
    const double map_height_m = 100.0;

    const double max_speed_mps = 10.0;

    // Read grayscale image and use it as a pt_map
    Mat img = imread("pt_map.png", 0);   // Read the file

    // We need to flip the image because we the x-axis is upwards
    img = flip(img);

    // Binarization, white to be 0
    Mat dest;
    threshold(img, dest, 127, 1, CV_THRESH_BINARY_INV);
    dest.convertTo(dest, CV_64FC1);

    auto map = make_shared<pt_map>(dest, map_width_s, map_height_m);
    auto car = make_shared<vehicle_model>(map);

    double step_p = 1.0;
    double step_v = 1.0;
    double step_t = 1.0;

    DPClass<vehicle_model> dp(car, step_t, step_p, step_v);

    auto start = chrono::high_resolution_clock::now();
    for(double t=map_width_s - step_t; t >= 0.0; t -=step_t)
    {
        size_t job_id = 0;
        vector<thread> jobs;
        for(double v = -1.0; v < max_speed_mps; v += step_v)
        {
            jobs.emplace_back(value_iteration,
                              job_id++,
                              &dp,
                              t,
                              map_height_m, 0, step_p,
                              v + step_v / 2, v, step_v);
        }
        for(auto & job: jobs)
            job.join();
    }

    auto finish = chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = finish - start;
    cout << "Optimization time: " << duration.count() << " s" << endl;

    auto value_t = dp.get_all_value();
    normalize(value_t, value_t, 0, 1, NORM_MINMAX, -1, Mat());

    auto policy_t = dp.get_all_policy();
    normalize(policy_t, policy_t, 0, 1, NORM_MINMAX, -1, Mat());

    namedWindow( "PT Map", WINDOW_NORMAL);
    imshow( "PT Map", flip(img));

    namedWindow( "Display Value", WINDOW_NORMAL);
    imshow( "Display Value", for_show(value_t));

    namedWindow( "Display Policy", WINDOW_NORMAL);
    imshow( "Display Policy", for_show(policy_t));

    // show optimal trajectory on pt_map

    pt_map overlay_map(img * 0.5, map_width_s, map_height_m);

    state x = {50.0, 0.0};
    double sim_dt = 0.5;
    cout << "time\tcontrol\n";
    for(double time = 0; time < 10.0; time += sim_dt)
    {
        overlay_map.draw_circle(x[0], time);
        const auto u = dp.policy_get(x, time);
        cout << time << "\t" << u[0] << endl;
        x = car->dynamics(x, u, sim_dt);
    }

    namedWindow( "Display Trajectory", WINDOW_NORMAL);
    imshow( "Display Trajectory", flip(overlay_map.get_value()));

    cout << "Press any key to exit." << endl;
    waitKey(0);

    return 0;
}

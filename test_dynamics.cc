#include "dynamics.hh"
#include "dp.hh"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main()
{
    const double map_width_s = 10.0;
    const double map_height_m = 100.0;
    Mat img = imread("pt_map.png", 0);   // Read the file
    img.convertTo(img, CV_64FC1);
    auto map = make_shared<pt_map>(img, map_width_s, map_height_m);
    vehicle_model car(map);
    // test pt_map

    for(double position = map_height_m; position >=0; position -= 1)
    {
        for(double time = 0; time < map_width_s; time += 1)
        {
            cout << map->get(time, position) << " ";
        }
        cout << endl;
    }

    return 0;
}

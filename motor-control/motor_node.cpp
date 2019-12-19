#include <ros/ros.h>
#include <pigpiod_if2.h>
#include <ctime>
#include <geometry_msgs/Point.h>
#include <math.h>
#include <unistd.h>
#include <std_msgs/String.h>


//#include <wiringPi.h>
//#include <linux/timer.h>
//#include <include/linux/jiffies.h>

#define motor_DIR1 26
#define motor_PWM1 12
#define motor_ENA1 6
#define motor_DIR2 19
#define motor_PWM2 13
#define motor_ENA2 22

using namespace std;

int PWM_limit;
void Interrupt1(int pi, unsigned user_gpio, unsigned level, uint32_t tick);
void Interrupt2(int pi, unsigned user_gpio, unsigned level, uint32_t tick);
int Limit_Function(int pwm);

volatile int EncoderCounter1;
volatile int EncoderCounter2;
bool switch_direction;
int Theta_Distance_Flag;
volatile clock_t Prev_time = 0;

static inline long mytime(){
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec * 1000 + tv.tv_usec / 1000);
}

class DcMotorForRaspberryPi
{
private:

public:
  int pinum;
  int motor_ENA;
  int motor_DIR;
  int motor_PWM;
  int PWM_range;
  int PWM_frequency;
  int current_PWM;
  bool current_Direction;
  int acceleration;
  void Motor_Controller(bool direction, int pwm);
  void Accel_Controller(bool direction, int desired_pwm);
  DcMotorForRaspberryPi(){}
  DcMotorForRaspberryPi(int motor_dir, int motor_pwm, int motor_ena)
  {
    pinum=pigpio_start(NULL, NULL);
    if(pinum<0)
    {
      ROS_INFO("Setup failed");
      ROS_INFO("pinum is %d", pinum);
    }
    motor_DIR = motor_dir;
    motor_PWM = motor_pwm;
    motor_ENA = motor_ena;
    PWM_range = 512;
    PWM_frequency = 40000; 

    set_mode(pinum, motor_dir, PI_OUTPUT);
    set_mode(pinum, motor_pwm, PI_OUTPUT);
    set_mode(pinum, motor_ena, PI_INPUT);

    set_PWM_range(pinum, motor_pwm, PWM_range);
    set_PWM_frequency(pinum, motor_pwm, PWM_frequency);
    gpio_write(pinum, motor_DIR, PI_LOW);
    set_PWM_dutycycle(pinum, motor_PWM, 0);
    
    current_PWM = 0;
    current_Direction = true;
    acceleration = 20;
    ROS_INFO("Setup Fin");
  }
};

void DcMotorForRaspberryPi::Motor_Controller(bool direction, int pwm)
{
  if(direction == true) //CW
  {
    gpio_write(pinum, motor_DIR, PI_LOW);
    set_PWM_dutycycle(pinum, motor_PWM, pwm);
    current_PWM = pwm;
    current_Direction = true;
  }
  else //CCW
  {
    gpio_write(pinum, motor_DIR, PI_HIGH);
    set_PWM_dutycycle(pinum, motor_PWM, pwm);
    current_PWM = pwm;
    current_Direction = false;
  }
}
void DcMotorForRaspberryPi::Accel_Controller(bool direction, int desired_pwm)
{
  int local_PWM;
  if(desired_pwm > current_PWM)
  {
    local_PWM = current_PWM + acceleration;
    Motor_Controller(direction, local_PWM);
  }
  else if(desired_pwm < current_PWM)
  {
    local_PWM = current_PWM - acceleration;
    Motor_Controller(direction, local_PWM);
  }
  else
  {
    local_PWM = current_PWM;
    Motor_Controller(direction, local_PWM);
  }
  //ROS_INFO("Current_PWM is %d", current_PWM);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////
DcMotorForRaspberryPi motor1 = DcMotorForRaspberryPi(motor_DIR1, motor_PWM1, motor_ENA1);
DcMotorForRaspberryPi motor2 = DcMotorForRaspberryPi(motor_DIR2, motor_PWM2, motor_ENA2);

void Initialize()
{
  PWM_limit = 512;
  EncoderCounter1 = 0;
  EncoderCounter2 = 0;
  callback(motor1.pinum, motor1.motor_ENA, FALLING_EDGE, Interrupt1);
  callback(motor1.pinum, motor2.motor_ENA, FALLING_EDGE, Interrupt2);

  switch_direction = true;
  Theta_Distance_Flag = 0;
  ROS_INFO("Initialize Complete");
}
void Interrupt1(int pi, unsigned user_gpio, unsigned level, uint32_t tick)
{
  EncoderCounter1 ++;
//  ROS_INFO("Interrupt1 is %d", EncoderCounter1);
}
void Interrupt2(int pi, unsigned user_gpio, unsigned level, uint32_t tick)
{
  EncoderCounter2 ++;
 // ROS_INFO("Interrupt2 is %d", EncoderCounter2);
}

int Limit_Function(int pwm)
{
  int output;
  if(pwm > PWM_limit)output = PWM_limit;
  else if(pwm < 0)output = 0;
  else output = pwm;
  return output; 
}

void Switch_Turn_Example(int PWM1, int PWM2)
{
  int local_PWM1 = Limit_Function(PWM1);
  int local_PWM2 = Limit_Function(PWM2);
  if(switch_direction == true)
  {
    motor1.Motor_Controller(switch_direction, local_PWM1);
    motor2.Motor_Controller(switch_direction, local_PWM2);
    switch_direction = false;
    ROS_INFO("true");
  }
  else
  {
    motor1.Motor_Controller(switch_direction, local_PWM1);
    motor2.Motor_Controller(switch_direction, local_PWM2);
    switch_direction = true;
    ROS_INFO("false");
  }
  ROS_INFO("Encoder A1 is %d", EncoderCounter1);
  ROS_INFO("Encoder A2 is %d", EncoderCounter2);
}

void Theta_Turn(float Theta, int PWM)
{
  double local_encoder;
  int local_PWM = Limit_Function(PWM);
  if(Theta_Distance_Flag == 1)
  {
      EncoderCounter1 = 0;
      EncoderCounter2 = 0;
      Theta_Distance_Flag = 2;
  }
  if(Theta > 0)
  {
    local_encoder = Theta; //(Round_Encoder/360)*(Robot_Round/Wheel_Round)
    motor1.Motor_Controller(true, local_PWM);
    motor2.Motor_Controller(true, local_PWM);
    //motor1.Accel_Controller(true, local_PWM);
    //motor2.Accel_Controller(true, local_PWM);
  }
  else
  {
    local_encoder = -Theta; //(Round_Encoder/360)*(Robot_Round/Wheel_Round)
    motor1.Motor_Controller(false, local_PWM);
    motor2.Motor_Controller(false, local_PWM);
    //motor1.Accel_Controller(false, local_PWM);
    //motor2.Accel_Controller(false, local_PWM);
  }

  if(EncoderCounter1 > local_encoder)
  {
    //ROS_INFO("Encoder A1 is %d", EncoderCounter1);
    //ROS_INFO("Encoder A2 is %d", EncoderCounter2);
    EncoderCounter1 = 0;
    EncoderCounter2 = 0;
    motor1.Motor_Controller(true, 0);
    motor2.Motor_Controller(true, 0);
    //motor1.Motor_Controller(true, 0);
    //motor2.Motor_Controller(true, 0);
    Theta_Distance_Flag = 3;
  }
}
void Distance_Go(float Distance, int PWM)
{
  float local_encoder = Distance; //(Round_Encoder*Distance)/Wheel_Round
  int local_PWM = Limit_Function(PWM);
  bool Direction = 1;
  if(Distance < 0)
  {
    Direction = 0;
    local_encoder = -local_encoder;
  }
  if(Theta_Distance_Flag == 3)
  {
      EncoderCounter1 = 0;
      EncoderCounter2 = 0;
      Theta_Distance_Flag = 4;
  }

  if(EncoderCounter1 < local_encoder)
  {
    if(Direction==1)
    {
      motor1.Motor_Controller(false, local_PWM);
      motor2.Motor_Controller(true, local_PWM);
      //motor1.Accel_Controller(false, local_PWM);
      //motor2.Accel_Controller(true, local_PWM);
    }
    else
    {
      motor1.Motor_Controller(true, local_PWM);
      motor2.Motor_Controller(false, local_PWM);
      //motor1.Motor_Controller(true, local_PWM);
      //motor2.Motor_Controller(false, local_PWM);
    }
  }
  else
  {
    //ROS_INFO("Encoder A1 is %d", EncoderCounter1);
    //ROS_INFO("Encoder A2 is %d", EncoderCounter2);
    EncoderCounter1 = 0;
    EncoderCounter2 = 0;
    motor1.Motor_Controller(true, 0);
    motor2.Motor_Controller(true, 0);
    //motor1.Accel_Controller(true, 0);
    //motor2.Accel_Controller(true, 0);
    Theta_Distance_Flag = 0;
  }
}


void CountRPM(double &Right_motr, double &Left_motr){
 
  static int Prev_EC1 = 0;
  static int Prev_EC2 = 0;
  float RPM1;
  float RPM2;
  

  int Diff_EC1 = EncoderCounter1-Prev_EC1;
  int Diff_EC2 = EncoderCounter2-Prev_EC2;
  clock_t time = mytime();

  double Diff_time =(double)(time - Prev_time);

  double ECsecond = Diff_time/1000; // Second

  cout <<"ECsecond " << ECsecond << endl;

  RPM1 = (float)(EncoderCounter1-Prev_EC1)/ECsecond * 60 * 1/348.3;
  RPM2 = (float)(EncoderCounter2-Prev_EC2)/ECsecond * 60 * 1/348.3;

  cout << "RPM Motor 1 : " << RPM1 << endl;
  cout << "RPM Motor 2 : " << RPM2 << endl;

  Right_motr = RPM1;
  Left_motr = RPM2;
 
  Prev_EC1 = EncoderCounter1;
  Prev_EC2 = EncoderCounter2;

  Prev_time = mytime();

}


void Theta_Distance(float Theta, int Turn_PWM, float Distance, int Go_PWM)
{
  if(Theta_Distance_Flag == 0)
  {
    Theta_Distance_Flag = 1;
  }
  else if(Theta_Distance_Flag == 1 || Theta_Distance_Flag == 2)
  {
    Theta_Turn(Theta, Turn_PWM);
  }
  else if(Theta_Distance_Flag == 3 || Theta_Distance_Flag == 4)
  {
    Distance_Go(Distance, Go_PWM);
  }
}


void PIDcontrol(int targetRPM){
//Target is RPM
  
  bool Direc;
  int TargetRPM = targetRPM;
  float Kp = 0.5;
  float Ki = 0.5;
  float Kd = 0.2;

  static double NOW_RPM2=0, NOW_RPM=0;

  int error=0;  

  
  static float Prev_error=0;
  
  //static NOW_RPM;

  CountRPM(NOW_RPM,NOW_RPM2);
  error = int(TargetRPM - int(NOW_RPM));
  float PC, DC;
  static float IC = 0;
  float Time = 0.1;
  int PID;
  float RPM = 0;
 // error = int(TargetRPM - int(NOW_RPM));

  PC = Kp * error;
  IC += Ki * error * Time;
  DC = Kd * (error - Prev_error)/Time;

 // CountRPM(NOW_RPM,NOW_RPM2);
  PID =int(PC + IC + DC);

  printf("PID :  %d, err : %d \n", PID,error);
 

  if(PID > 512) PID = 512;
 // if(PID > 0)// Direc = false;
  if (PID<0) {
//    Direc  = true;
   PID = -PID;
 }
  
  int Input_PWM = PID; //* 512/170;
  motor1.Motor_Controller(switch_direction,Input_PWM);


 
  Prev_error = error;
}

void RightMot(int motrspd){
  int target = (motrspd * 512)/100;
  bool wheel;
  
  if(target > 0) {
    wheel = false;
  }else{
    wheel = true;
    target = -target;
  }

  if(target >512) target = 512;
  motor1.Motor_Controller(wheel, target);

}

void LeftMot(int motrspd){
  int target = (motrspd * 512)/100;
  bool wheel;

  if(target > 0){
    wheel = true;
    
  }else{
    wheel = false;
    target = -target;
  }
  if(target >512) target = 512;

  motor2.Motor_Controller(wheel,target);
}

void Stop(){
 
  LeftMot(0);
  RightMot(0);

}



void Velocity(int motor1spd, int motor2spd){

  int motor1_PWM = (motor1spd*512)/100;
  int motor2_PWM = (motor2spd*512)/100;

  int local_PWM1 = Limit_Function(motor1_PWM);
  int local_PWM2 = Limit_Function(motor2_PWM);

  if(motor1spd > 0) motor1.Accel_Controller(false, local_PWM1);
  else motor1.Accel_Controller(true, local_PWM1);
  if(motor2spd > 0) motor2.Accel_Controller(true, local_PWM2);
  else motor2.Accel_Controller(false, local_PWM2);

   // motor1.Accel_Controller(true, motor1_PWM);
   // motor2.Accel_Controller(false, motor2_PWM);
}


int flag = 1; //Motor run?
int change =0; //change?
int prev_flag = 1;
string abc = "no";

void XYZCallback(const geometry_msgs::Point data){
 //cout>> data->x >> endl;
 //cout>>data->y>>endl;
 //cout>>data->z>>endl;
 
  if (flag == 0 ){
    Stop();
    prev_flag = 0;
    return;

  }

  if (data.z > 0 && data.z < 1){
    printf("Person is Near\n");
    printf("STOP z : %lf\n", data.z);
   // Velocity(0, 0);
    
    Stop();

  }else if (data.z < 0 && data.z >-1.5){ //All Point is 0 mm
    printf("[NOW Z] -1\n");
    Stop();
    //Switch_Turn_Example(200,200);
    RightMot(30);
    LeftMot(-30);
    sleep(5);
    Stop();

  }else if (data.z > -3 && data.z < -1.5){ //Obstacle Detected
    printf("Obstacle Detecting \n");
    //Stop();
    RightMot(-30);
    LeftMot(30);
    sleep(1);
   // Stop();
   // sleep(1);
   // RightMot(-50);
  //  LeftMot(50);
  //  sleep(3);
  //  Stop();
  }

  else{
    if (data.x < 320-50){
      printf("GO LEFT\n");
      LeftMot(int(data.x-320.0)/15+30);
      RightMot(int(320.0-data.x)/15+30);
    ///// Velocity(30 + (int)(320.0-data.x)/10,30); // set speed proportionally
    // Velocity(30+ (int)((data.x - 300.0)/5),30);
     // Velocity(50,40); // test
    }
    else if(data.x>320+50){
      printf("GO RIGHT\n");
      ////Velocity(30,30 + (int)((data.x-320.0)/10)); // set speed proportionally
      LeftMot(int(data.x-320.0)/15+30);
      RightMot(int(320.0 - data.x)/15+30);

      // Velocity(30, 30 + (int)((300.0 - data.x)/5)); 
      // Velocity(40, 50); 
    }
    else{
      Velocity(40,40);
      printf("GO STAIGHT");
    }


  }
}


void StopMotorCallback(const std_msgs::String::ConstPtr& msg){
  string S = msg->data.c_str();
  string STOP = "STOP";
  string GO = "GO";

  if (S.compare(STOP)==0){
    cout << "STOP!!!!!!!!!! "<< endl;
    flag = 0;
  }

  else if(S.compare(GO)==0){
    cout <<" GO GO "<< endl;

    flag =1;
  }

}



int main(int argc, char** argv)
{
  double RightM, LeftM;

  ros::init(argc, argv, "Motor_node");
  ros::NodeHandle nh;

  Initialize();
  ros::Subscriber motrctrl = nh.subscribe("XYZ_topic", 10, XYZCallback);
  ros::Rate loop_rate(10);
  ros::Subscriber Detector = nh.subscribe("stop_motor", 10, StopMotorCallback);
    
  //LeftMot(-50);
  while(ros::ok())
  {
   
   // LeftMot(50);
   // Switch_Turn_Example(100, 100);
   // Theta_Distance(400,100,0,110);
    
   // motor1.Accel_Controller(true, 122);
   // motor2.Accel_Controller(false, 512);
   // Velocity(50, 50);
   // PIDcontrol(100);
  

   // motor1.Motor_Controller(false,50);

    ros::spinOnce();
    //CountRPM(RightM, LeftM);
    loop_rate.sleep(); 
  }
  motor1.Motor_Controller(true, 0);
  motor2.Motor_Controller(true, 0);
 
  return 0;
}

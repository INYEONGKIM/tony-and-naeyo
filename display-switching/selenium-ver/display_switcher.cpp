#include <stdlib.h>
#include <stdio.h>

int main(int argc, char *argv[]){
    int signum;

    system("ps | grep app.py | awk 'NR<2{print $1}' | xargs kill -9");
    
    while(true){
        scanf("%d", &signum);
        if (signum==1){
            // to main
            system("ps | grep firefoxOpener | awk 'NR<2{print $1}' | xargs kill -30");
        }else if(signum==2){
            // to map
            system("ps | grep firefoxOpener | awk 'NR<2{print $1}' | xargs kill -31");
        }else if(signum==3){
            // to map
            system("ps | grep firefoxOpener | awk 'NR<2{print $1}' | xargs kill -29");
        }else {
            printf("^^ㅗ\n");
            break;
        }
    }
    return 0;
}

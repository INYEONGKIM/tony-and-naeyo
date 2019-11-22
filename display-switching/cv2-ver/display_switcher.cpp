#include <stdlib.h>
#include <stdio.h>

int main(int argc, char *argv[]){
    int signum;

    while(true){
        scanf("%d", &signum);
        if (signum==1){
            // to happy
            system("ps | grep cv-display-switcher.py | awk 'NR<2{print $1}'
					| xargs kill -10");
        }else if(signum==2){
            // to normal
            system("ps | grep cv-display-switcher.py | awk 'NR<2{print $1}'
					| xargs kill -12");
        }else if(signum==3){
            // to map
            // system("ps | grep cv-display-switcher.py | awk 'NR<2{print $1}' | xargs kill -29");
        }else {
            printf("^^ㅗ\n");
            break;
        }
    }
    return 0;
}

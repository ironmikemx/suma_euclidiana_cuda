#include <stdio.h>

main() {

   FILE *fp;
   char buff[255];
  int temp;

   fp = fopen("test.txt", "r");
   fscanf(fp, "%s", buff);
   printf("1 : %s\n", buff );

   fscanf(fp, "%d", &temp);
   temp = temp * 10;
   printf("1 : %d\n", temp );

   fclose(fp);

}

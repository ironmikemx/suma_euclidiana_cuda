#!/usr/bin/perl
use strict;
use warnings;

my $timer1;
my $timer2;
my $cmd;
#for (my $u = 0; $u < 944; $u++) {
#  for (my $m = 0; $m < 1682; $m++) {
#    for (my $i = 0; $i < 944; $i++) { 
for (my $u = 10; $u < 15; $u++) {
  for (my $m = 10; $m < 12; $m++) {
    for (my $i = 0; $i < 3; $i++) {
      
      $cmd  = "( time ./suma $u $m $i 0 0 > /dev/null) 2>&1 | grep real ";
      print $cmd;
      $timer1 = `$cmd`;
      print "R $timer1 R";
      chomp($timer1);
      $timer1 =~ s/s//g;
      $cmd  = "( time ./suma $u $m $i 1 0 > /dev/null) 2>&1 | grep real | awk '{print \$2}'";
      $timer2 = `$cmd`;
      chomp($timer2);
      $timer2 =~ s/s//g;
      print "$u $m $i $timer1 $timer2\n"
    }
   }
}

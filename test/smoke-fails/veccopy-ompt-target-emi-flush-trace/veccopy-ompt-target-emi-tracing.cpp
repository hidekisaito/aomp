#include <assert.h>
#include <omp.h>
#include <stdio.h>

#include "callbacks.h"

// Map of devices traced
DeviceMapPtr_t DeviceMapPtr;

int main() {
  int N = 100000;

  int a[N];
  int b[N];

  int i;

  for (i = 0; i < N; i++)
    a[i] = 0;

  for (i = 0; i < N; i++)
    b[i] = i;

#pragma omp target parallel for
  {
    for (int j = 0; j < N; j++)
      a[j] = b[j];
  }

#pragma omp target teams distribute parallel for
  {
    for (int j = 0; j < N; j++)
      a[j] = b[j];
  }

  int rc = 0;
  for (i = 0; i < N; i++)
    if (a[i] != b[i]) {
      rc++;
      printf("Wrong value: a[%d]=%d\n", i, a[i]);
    }

  if (!rc)
    printf("Success\n");

  return rc;
}

// clang-format off

/// CHECK-NOT: host_op_id=0x0

/// CHECK-DAG: Allocated {{[0-9]+}} bytes at [[ADDRX_01:0x[0-f]+]] in buffer request callback
/// CHECK-DAG: Allocated {{[0-9]+}} bytes at [[ADDRX_02:0x[0-f]+]] in buffer request callback
/// CHECK-DAG: Allocated {{[0-9]+}} bytes at [[ADDRX_03:0x[0-f]+]] in buffer request callback
/// CHECK-DAG: Allocated {{[0-9]+}} bytes at [[ADDRX_04:0x[0-f]+]] in buffer request callback
/// CHECK-DAG: Allocated {{[0-9]+}} bytes at [[ADDRX_05:0x[0-f]+]] in buffer request callback
/// CHECK-DAG: Allocated {{[0-9]+}} bytes at [[ADDRX_06:0x[0-f]+]] in buffer request callback
/// CHECK-DAG: Allocated {{[0-9]+}} bytes at [[ADDRX_07:0x[0-f]+]] in buffer request callback
/// CHECK-DAG: Allocated {{[0-9]+}} bytes at [[ADDRX_08:0x[0-f]+]] in buffer request callback
/// CHECK-DAG: Allocated {{[0-9]+}} bytes at [[ADDRX_09:0x[0-f]+]] in buffer request callback
/// CHECK-DAG: Allocated {{[0-9]+}} bytes at [[ADDRX_10:0x[0-f]+]] in buffer request callback
/// CHECK-DAG: Allocated {{[0-9]+}} bytes at [[ADDRX_11:0x[0-f]+]] in buffer request callback

/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_01]] {{[0-9]+}} [[ADDRX_01]] {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_02]] {{[0-9]+}} [[ADDRX_02]] {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_03]] {{[0-9]+}} [[ADDRX_03]] {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_04]] {{[0-9]+}} [[ADDRX_04]] {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_05]] {{[0-9]+}} [[ADDRX_05]] {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_06]] {{[0-9]+}} [[ADDRX_06]] {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_07]] {{[0-9]+}} [[ADDRX_07]] {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_08]] {{[0-9]+}} [[ADDRX_08]] {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_09]] {{[0-9]+}} [[ADDRX_09]] {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_10]] {{[0-9]+}} [[ADDRX_10]] {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_11]] {{[0-9]+}} [[ADDRX_11]] {{[0-9]+}}

// Note: Split checks for record address and content. That way we do not imply
//       any order. Records 01-06 and 12-17 occur interleaved and belong to the
//       first target region. 07-11 occur interleaved with 18-22 and belong to
//       the second target region.
/// CHECK-DAG: rec=[[ADDRX_01]]
/// CHECK-DAG: rec=[[ADDRX_02]]
/// CHECK-DAG: rec=[[ADDRX_03]]
/// CHECK-DAG: rec=[[ADDRX_04]]
/// CHECK-DAG: rec=[[ADDRX_05]]
/// CHECK-DAG: rec=[[ADDRX_06]]
/// CHECK-DAG: rec=[[ADDRX_07]]
/// CHECK-DAG: rec=[[ADDRX_08]]
/// CHECK-DAG: rec=[[ADDRX_09]]
/// CHECK-DAG: rec=[[ADDRX_10]]
/// CHECK-DAG: rec=[[ADDRX_11]]

/// CHECK-DAG: type=8 (Target task) {{.+}} kind=1 endpoint=1
/// CHECK-DAG: type=9 (Target data op) {{.+}} optype=1
/// CHECK-DAG: type=9 (Target data op) {{.+}} optype=2
/// CHECK-DAG: type=9 (Target data op) {{.+}} optype=1
/// CHECK-DAG: type=9 (Target data op) {{.+}} optype=2
/// CHECK-DAG: type=10 (Target kernel) {{.+}} requested_num_teams=1
/// CHECK-DAG: type=9 (Target data op) {{.+}} optype=3
/// CHECK-DAG: type=9 (Target data op) {{.+}} optype=3
/// CHECK-DAG: type=9 (Target data op) {{.+}} optype=4
/// CHECK-DAG: type=9 (Target data op) {{.+}} optype=4
/// CHECK-DAG: type=8 (Target task) {{.+}} kind=1 endpoint=2

// Note: These addresses will only occur once. They are only captured to
//       indicate their existence.
/// CHECK-DAG: rec=[[ADDRX_12:0x[0-f]+]]
/// CHECK-DAG: rec=[[ADDRX_13:0x[0-f]+]]
/// CHECK-DAG: rec=[[ADDRX_14:0x[0-f]+]]
/// CHECK-DAG: rec=[[ADDRX_15:0x[0-f]+]]
/// CHECK-DAG: rec=[[ADDRX_16:0x[0-f]+]]
/// CHECK-DAG: rec=[[ADDRX_17:0x[0-f]+]]
/// CHECK-DAG: rec=[[ADDRX_18:0x[0-f]+]]
/// CHECK-DAG: rec=[[ADDRX_19:0x[0-f]+]]
/// CHECK-DAG: rec=[[ADDRX_20:0x[0-f]+]]
/// CHECK-DAG: rec=[[ADDRX_21:0x[0-f]+]]
/// CHECK-DAG: rec=[[ADDRX_22:0x[0-f]+]]

/// CHECK-DAG: type=8 (Target task) {{.+}} kind=1 endpoint=1
/// CHECK-DAG: type=9 (Target data op) {{.+}} optype=1
/// CHECK-DAG: type=9 (Target data op) {{.+}} optype=2
/// CHECK-DAG: type=9 (Target data op) {{.+}} optype=1
/// CHECK-DAG: type=9 (Target data op) {{.+}} optype=2
/// CHECK-DAG: type=10 (Target kernel) {{.+}} requested_num_teams=0
/// CHECK-DAG: type=9 (Target data op) {{.+}} optype=3
/// CHECK-DAG: type=9 (Target data op) {{.+}} optype=3
/// CHECK-DAG: type=9 (Target data op) {{.+}} optype=4
/// CHECK-DAG: type=9 (Target data op) {{.+}} optype=4
/// CHECK-DAG: type=8 (Target task) {{.+}} kind=1 endpoint=2

// Note: ADDRX_11 may not trigger a final callback.
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_01]] {{[0-9]+}} (nil) {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_02]] {{[0-9]+}} (nil) {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_03]] {{[0-9]+}} (nil) {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_04]] {{[0-9]+}} (nil) {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_05]] {{[0-9]+}} (nil) {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_06]] {{[0-9]+}} (nil) {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_07]] {{[0-9]+}} (nil) {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_08]] {{[0-9]+}} (nil) {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_09]] {{[0-9]+}} (nil) {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_10]] {{[0-9]+}} (nil) {{[0-9]+}}

// Note: ADDRX_11 may not be deallocated.
/// CHECK-DAG: Deallocated [[ADDRX_01]]
/// CHECK-DAG: Deallocated [[ADDRX_02]]
/// CHECK-DAG: Deallocated [[ADDRX_03]]
/// CHECK-DAG: Deallocated [[ADDRX_04]]
/// CHECK-DAG: Deallocated [[ADDRX_05]]
/// CHECK-DAG: Deallocated [[ADDRX_06]]
/// CHECK-DAG: Deallocated [[ADDRX_07]]
/// CHECK-DAG: Deallocated [[ADDRX_08]]
/// CHECK-DAG: Deallocated [[ADDRX_09]]
/// CHECK-DAG: Deallocated [[ADDRX_10]]

/// CHECK-DAG: Success

/// CHECK-NOT: rec=
/// CHECK-NOT: host_op_id=0x0

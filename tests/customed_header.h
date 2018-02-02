
static inline void func(CField *f, float *lon, float *dt)
{
  float (*data)[f->xdim] = (float (*)[f->xdim]) f->data;
  float u = data[2][1];
  *lon += u * *dt;
}

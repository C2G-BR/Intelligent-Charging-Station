import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { map, retry } from 'rxjs/operators';

@Injectable({
  providedIn: 'root'
})
export class IcsServiceService {

  constructor(private httpClient: HttpClient) { }

  public generateData(api) {
    return this.httpClient.get(api + 'generate/').pipe(
      retry(2),
      map((data2: any) => {
        return data2
      })
    );
  }
}

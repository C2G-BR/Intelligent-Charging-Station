import { TestBed } from '@angular/core/testing';

import { IcsServiceService } from './ics-service.service';

describe('IcsServiceService', () => {
  let service: IcsServiceService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(IcsServiceService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
